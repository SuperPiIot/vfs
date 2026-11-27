// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package most_test

import (
	"bytes"
	"context"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"golang.org/x/crypto/chacha20poly1305"

	"github.com/hanzoai/vfs/internal/blake3"
	"github.com/hanzoai/vfs/internal/storetest"
	"github.com/hanzoai/vfs/pkg/backend"
	"github.com/hanzoai/vfs/pkg/most"
)

// aeadCipher is an authenticated ChaCha20-Poly1305 Cipher for pkg/most unit tests
// without importing package vfs.
type aeadCipher struct {
	id   string
	aead cipher.AEAD
}

func newTestCipher(t testing.TB, seed string) most.Cipher {
	t.Helper()
	sum := sha256.Sum256([]byte(seed))
	aead, err := chacha20poly1305.NewX(sum[:])
	if err != nil {
		t.Fatal(err)
	}
	return &aeadCipher{
		id:   hex.EncodeToString(sum[:8]),
		aead: aead,
	}
}

func (c *aeadCipher) Seal(plain []byte) ([]byte, string, error) {
	nonce := make([]byte, c.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, "", err
	}
	sealed := c.aead.Seal(nonce, nonce, plain, []byte(c.id))
	return sealed, c.id, nil
}

func (c *aeadCipher) Open(sealed []byte, keyID string) ([]byte, error) {
	if keyID != c.id {
		return nil, fmt.Errorf("unknown key %s", keyID)
	}
	ns := c.aead.NonceSize()
	if len(sealed) < ns {
		return nil, errors.New("short ciphertext")
	}
	return c.aead.Open(nil, sealed[:ns], sealed[ns:], []byte(c.id))
}

func testPayload(seed byte, size int) []byte {
	b := make([]byte, size)
	for i := range b {
		b[i] = seed ^ byte(i*31+7)
	}
	return b
}

func TestSelectiveMirrorOneHitScanNoPollutionAndZeroDemotion(t *testing.T) {
	ctx := context.Background()
	mem := storetest.NewMem()
	nvmeDir := t.TempDir()

	const objSize = 4096
	eng, err := most.NewEngine(mem, most.Config{
		Policy:         most.PolicySelectiveMirror,
		NVMeDir:        nvmeDir,
		TotalRAMBytes:  4 * objSize,
		TotalNVMeBytes: 128 * objSize,
		HotThreshold:   2.0,
		SharedPrefix:   "tc",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	cA := newTestCipher(t, "tenant-a")
	if err := eng.RegisterTenant("tenant-a", "vol-a", cA, 4*objSize, 128*objSize); err != nil {
		t.Fatal(err)
	}

	// Publish 8 shared immutable toolchain blobs + 40 cold scan blobs.
	for i := range 8 {
		if _, err := eng.PublishShared(ctx, fmt.Sprintf("hot-%02d", i), testPayload(byte(i+1), objSize)); err != nil {
			t.Fatal(err)
		}
	}
	for i := range 40 {
		if _, err := eng.PublishShared(ctx, fmt.Sprintf("cold-%03d", i), testPayload(byte(i+50), objSize)); err != nil {
			t.Fatal(err)
		}
	}

	// 1st pass over hot-00..hot-03: fetches from backend into NVMe ONLY (0 RAM writes).
	for i := range 4 {
		got, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, fmt.Sprintf("hot-%02d", i), backend.Unbounded)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, testPayload(byte(i+1), objSize)) {
			t.Fatalf("hot-%02d mismatch", i)
		}
	}
	st := eng.Stats()
	if st.BackendGets != 4 || st.MirrorPromotions != 0 || st.ResidentRAMBytes != 0 {
		t.Fatalf("after 1st hit expected NVMe-only residency: %+v", st)
	}
	if st.ResidentNVMeBytes != 4*objSize {
		t.Fatalf("ResidentNVMeBytes = %d, want %d", st.ResidentNVMeBytes, 4*objSize)
	}

	// 2nd pass over hot-00..hot-03: hits NVMe and mirrors into RAM (keeping NVMe intact).
	for i := range 4 {
		if _, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, fmt.Sprintf("hot-%02d", i), backend.Unbounded); err != nil {
			t.Fatal(err)
		}
	}
	st = eng.Stats()
	if st.NVMeHits != 4 || st.MirrorPromotions != 4 || st.ResidentRAMBytes != 4*objSize || st.TierMigrationWrites != 0 {
		t.Fatalf("after 2nd hit expected RAM mirrors with 0 demotion writes: %+v", st)
	}

	// Sequential 1-hit cold scan of 40 blobs: MUST NOT pollute RAM or evict hot-00..hot-03.
	for i := range 40 {
		if _, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, fmt.Sprintf("cold-%03d", i), backend.Unbounded); err != nil {
			t.Fatal(err)
		}
	}
	st = eng.Stats()
	if st.RAMEvictions != 0 || st.MirrorPromotions != 4 || st.TierMigrationWrites != 0 {
		t.Fatalf("one-hit scan polluted RAM: %+v", st)
	}

	// Re-read hot-00..hot-03: all 4 hit RAM directly (0 backend GETs, 0 NVMe hits).
	beforeRAMHits := st.RAMHits
	beforeGets := st.BackendGets
	for i := range 4 {
		if _, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, fmt.Sprintf("hot-%02d", i), backend.Unbounded); err != nil {
			t.Fatal(err)
		}
	}
	st = eng.Stats()
	if st.RAMHits-beforeRAMHits != 4 || st.BackendGets != beforeGets {
		t.Fatalf("expected 4 RAMHits after cold scan, got %+v", st)
	}

	// Now promote hot-04..hot-07 (2 hits each): evicts hot-00..hot-03 mirrors from RAM
	// with ZERO demotion write-back (NVMe copy stayed intact).
	for pass := range 2 {
		_ = pass
		for i := 4; i < 8; i++ {
			if _, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, fmt.Sprintf("hot-%02d", i), backend.Unbounded); err != nil {
				t.Fatal(err)
			}
		}
	}
	st = eng.Stats()
	if st.RAMEvictions != 4 || st.TierMigrationWrites != 0 {
		t.Fatalf("RAM mirror eviction triggered demotion writes: %+v", st)
	}

	// Re-reading evicted hot-00..hot-03 hits NVMe with 0 extra BackendGets!
	getsBefore := st.BackendGets
	nvmeBefore := st.NVMeHits
	for i := range 4 {
		if _, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, fmt.Sprintf("hot-%02d", i), backend.Unbounded); err != nil {
			t.Fatal(err)
		}
	}
	st = eng.Stats()
	if st.BackendGets != getsBefore || st.NVMeHits-nvmeBefore != 4 {
		t.Fatalf("expected 4 NVMeHits on evicted mirrors with 0 backend gets: %+v", st)
	}
}

func TestTierMigrationDemotionWriteAmplificationVsMOST(t *testing.T) {
	ctx := context.Background()
	const objSize = 4096

	runWorkload := func(policy most.Policy) most.Stats {
		mem := storetest.NewMem()
		eng, err := most.NewEngine(mem, most.Config{
			Policy:         policy,
			TotalRAMBytes:  4 * objSize,
			TotalNVMeBytes: 64 * objSize,
			HotThreshold:   2.0,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eng.Close()
		if err := eng.RegisterTenant("t1", "v1", newTestCipher(t, "k1"), 4*objSize, 64*objSize); err != nil {
			t.Fatal(err)
		}
		for i := range 8 {
			_, _ = eng.PublishShared(ctx, fmt.Sprintf("hot-%d", i), testPayload(byte(i), objSize))
		}
		for i := range 24 {
			_, _ = eng.PublishShared(ctx, fmt.Sprintf("scan-%d", i), testPayload(byte(i+100), objSize))
		}
		// Warm hot keys, interleave 1-hit scans, re-access hot keys.
		for round := range 3 {
			for i := range 4 {
				_, _ = eng.Get(ctx, "t1", most.ScopeSharedImmutable, fmt.Sprintf("hot-%d", i), backend.Unbounded)
			}
			for j := range 8 {
				_, _ = eng.Get(ctx, "t1", most.ScopeSharedImmutable, fmt.Sprintf("scan-%d", round*8+j), backend.Unbounded)
			}
		}
		return eng.Stats()
	}

	mostStats := runWorkload(most.PolicySelectiveMirror)
	migStats := runWorkload(most.PolicyTierMigration)

	if mostStats.TierMigrationWrites != 0 {
		t.Fatalf("MOST TierMigrationWrites = %d, want 0", mostStats.TierMigrationWrites)
	}
	if migStats.TierMigrationWrites == 0 {
		t.Fatalf("PolicyTierMigration expected > 0 demotion writes, got 0")
	}
	if mostStats.RAMHits <= migStats.RAMHits {
		t.Fatalf("expected MOST RAMHits (%d) > PolicyTierMigration RAMHits (%d)", mostStats.RAMHits, migStats.RAMHits)
	}
}

func TestHotnessDecayWithClock(t *testing.T) {
	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	mem := storetest.NewMem()
	eng, err := most.NewEngine(mem, most.Config{
		Policy:        most.PolicySelectiveMirror,
		TotalRAMBytes: 1 << 20,
		HotThreshold:  2.0,
		DecayHalfLife: 30 * time.Second,
		Clock:         func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	if err := eng.RegisterTenant("t", "v", newTestCipher(t, "k"), 1<<20, 1<<20); err != nil {
		t.Fatal(err)
	}
	_, _ = eng.PublishShared(ctx, "k1", []byte("data-1"))

	// 1st hit at t=0 -> score 1.0 (< 2.0), stays in NVMe.
	if _, err := eng.Get(ctx, "t", most.ScopeSharedImmutable, "k1", backend.Unbounded); err != nil {
		t.Fatal(err)
	}
	if eng.Stats().MirrorPromotions != 0 {
		t.Fatalf("promoted on 1st hit")
	}

	// Advance 30s (one half-life): prior 1.0 decays to 0.5, + 1.0 = 1.5 (< 2.0), NOT mirrored yet.
	now = now.Add(30 * time.Second)
	if _, err := eng.Get(ctx, "t", most.ScopeSharedImmutable, "k1", backend.Unbounded); err != nil {
		t.Fatal(err)
	}
	if eng.Stats().MirrorPromotions != 0 {
		t.Fatalf("decayed stale hit should not promote yet (score=%.2f)", eng.HotnessScore("t", most.ScopeSharedImmutable, "k1"))
	}

	// Immediate follow-up hit: 1.5 + 1.0 = 2.5 >= 2.0 -> mirrors to RAM!
	now = now.Add(10 * time.Millisecond)
	if _, err := eng.Get(ctx, "t", most.ScopeSharedImmutable, "k1", backend.Unbounded); err != nil {
		t.Fatal(err)
	}
	if eng.Stats().MirrorPromotions != 1 {
		t.Fatalf("MirrorPromotions = %d, want 1", eng.Stats().MirrorPromotions)
	}
}

func TestFailClosedSecuritySwapsAndTamper(t *testing.T) {
	ctx := context.Background()
	mem := storetest.NewMem()
	nvmeDir := t.TempDir()
	eng, err := most.NewEngine(mem, most.Config{
		Policy:  most.PolicySelectiveMirror,
		NVMeDir: nvmeDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	// 1. Unregistered tenant fails closed.
	if _, err := eng.Get(ctx, "ghost", most.ScopePrivate, "k", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("unregistered Get err = %v, want ErrTenantUnauthorized", err)
	}
	if _, err := eng.Get(ctx, "ghost", most.ScopeSharedImmutable, "s", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("unregistered shared Get err = %v, want ErrTenantUnauthorized", err)
	}

	// 2. Tenant without cipher fails closed on ScopePrivate.
	if err := eng.RegisterTenant("nociph", "vol-nc", nil, 1<<20, 1<<20); err != nil {
		t.Fatal(err)
	}
	if err := eng.Put(ctx, "nociph", most.ScopePrivate, "k", []byte("secret")); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("Put without cipher err = %v, want ErrTenantUnauthorized", err)
	}
	if _, err := eng.Get(ctx, "nociph", most.ScopePrivate, "k", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("Get without cipher err = %v, want ErrTenantUnauthorized", err)
	}

	// 3. Ciphertext swap across tenants, volumes, and keys even when sharing a Cipher!
	sharedCiph := newTestCipher(t, "shared-kms-key")
	if err := eng.RegisterTenant("tenant-a", "vol-1", sharedCiph, 1<<20, 1<<20); err != nil {
		t.Fatal(err)
	}
	if err := eng.RegisterTenant("tenant-b", "vol-2", sharedCiph, 1<<20, 1<<20); err != nil {
		t.Fatal(err)
	}

	secret := []byte("top-secret-agent-token-xyz")
	if err := eng.Put(ctx, "tenant-a", most.ScopePrivate, "k1", secret); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Get(ctx, "tenant-a", most.ScopePrivate, "k1", backend.Unbounded); err != nil {
		t.Fatal(err)
	}
	if err := eng.Put(ctx, "tenant-a", most.ScopePrivate, "k2", []byte("other")); err != nil {
		t.Fatal(err)
	}
	if err := eng.Put(ctx, "tenant-b", most.ScopePrivate, "k1", []byte("tenant-b-data")); err != nil {
		t.Fatal(err)
	}

	// Verify Rule 1: zero plaintext on backend or NVMe disk.
	for k, v := range mem.Dump() {
		if bytes.Contains(v, secret) {
			t.Fatalf("backend object %s contains plaintext!", k)
		}
	}
	_ = filepath.WalkDir(nvmeDir, func(path string, d os.DirEntry, werr error) error {
		if werr != nil || d.IsDir() {
			return nil
		}
		raw, _ := os.ReadFile(path)
		if bytes.Contains(raw, secret) {
			t.Fatalf("NVMe file %s contains plaintext!", path)
		}
		return nil
	})

	// Swap tenant-a's backend ciphertext into tenant-b's backend key and use a fresh Engine (cold NVMe)
	// so backend Get hits the swapped ciphertext under the shared Cipher -> must fail with ErrIntegrityViolation!
	var keyA, keyB string
	for _, k := range mem.Keys() {
		raw, _, _ := mem.Get(ctx, k, backend.Unbounded)
		// Distinguish by opening with fresh engine or finding tenant-a/k1 vs tenant-b/k1 via separate helper.
		_ = raw
	}
	// Deterministic key lookup using backend swap across all p/ keys: replace tenant-b's object with tenant-a's!
	swapMem := storetest.NewMem()
	engSwap, _ := most.NewEngine(swapMem, most.Config{Policy: most.PolicySelectiveMirror})
	defer engSwap.Close()
	_ = engSwap.RegisterTenant("ta", "va", sharedCiph, 1<<20, 1<<20)
	_ = engSwap.RegisterTenant("tb", "vb", sharedCiph, 1<<20, 1<<20)
	_ = engSwap.Put(ctx, "ta", most.ScopePrivate, "k1", secret)
	kA := swapMem.Keys()[0]
	cipherA, _, _ := swapMem.Get(ctx, kA, backend.Unbounded)
	_ = engSwap.Put(ctx, "tb", most.ScopePrivate, "k1", []byte("b-orig"))
	var kB string
	for _, k := range swapMem.Keys() {
		if k != kA {
			kB = k
		}
	}
	// Overwrite tb/k1 in backend with ta/k1's sealed frame, then read from a fresh cold Engine.
	_ = swapMem.Put(ctx, kB, cipherA)
	engCold, _ := most.NewEngine(swapMem, most.Config{Policy: most.PolicySelectiveMirror})
	defer engCold.Close()
	_ = engCold.RegisterTenant("tb", "vb", sharedCiph, 1<<20, 1<<20)
	if _, err := engCold.Get(ctx, "tb", most.ScopePrivate, "k1", backend.Unbounded); !errors.Is(err, most.ErrIntegrityViolation) {
		t.Fatalf("cross-tenant ciphertext swap err = %v, want ErrIntegrityViolation", err)
	}

	// Same-tenant different-key swap (k1 -> k2) must also fail closed with ErrIntegrityViolation!
	_ = engCold.RegisterTenant("ta", "va", sharedCiph, 1<<20, 1<<20)
	_ = engCold.Put(ctx, "ta", most.ScopePrivate, "k2", []byte("orig-k2"))
	var kA2 string
	for _, k := range swapMem.Keys() {
		if k != kA && k != kB {
			kA2 = k
		}
	}
	_ = swapMem.Put(ctx, kA2, cipherA)
	engCold2, _ := most.NewEngine(swapMem, most.Config{Policy: most.PolicySelectiveMirror})
	defer engCold2.Close()
	_ = engCold2.RegisterTenant("ta", "va", sharedCiph, 1<<20, 1<<20)
	if _, err := engCold2.Get(ctx, "ta", most.ScopePrivate, "k2", backend.Unbounded); !errors.Is(err, most.ErrIntegrityViolation) {
		t.Fatalf("cross-key ciphertext swap err = %v, want ErrIntegrityViolation", err)
	}

	// 4. Shared ACL + BLAKE3 tampering + read-only enforcement.
	if _, err := eng.PublishShared(ctx, "go/bin/compile", []byte("elf-binary-go")); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.PublishShared(ctx, "rust/bin/rustc", []byte("elf-binary-rust")); err != nil {
		t.Fatal(err)
	}
	if err := eng.SetSharedAccess("tenant-b", true, "go/"); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Get(ctx, "tenant-b", most.ScopeSharedImmutable, "go/bin/compile", backend.Unbounded); err != nil {
		t.Fatalf("authorized prefix denied: %v", err)
	}
	if _, err := eng.Get(ctx, "tenant-b", most.ScopeSharedImmutable, "rust/bin/rustc", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("unauthorized shared prefix err = %v, want ErrTenantUnauthorized", err)
	}
	if err := eng.Put(ctx, "tenant-a", most.ScopeSharedImmutable, "go/bin/compile", []byte("evil")); !errors.Is(err, most.ErrReadOnly) {
		t.Fatalf("shared Put err = %v, want ErrReadOnly", err)
	}

	// Tamper with NVMe shared cache file on disk -> fail closed with ErrIntegrityViolation!
	var sharedFile string
	_ = filepath.WalkDir(filepath.Join(nvmeDir, "shared"), func(path string, d os.DirEntry, werr error) error {
		if werr == nil && !d.IsDir() {
			sharedFile = path
		}
		return nil
	})
	if sharedFile == "" {
		t.Fatal("expected shared file in NVMe cache")
	}
	rawShared, err := os.ReadFile(sharedFile)
	if err != nil {
		t.Fatal(err)
	}
	rawShared[len(rawShared)-1] ^= 0xff
	if err := os.WriteFile(sharedFile, rawShared, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Get(ctx, "tenant-a", most.ScopeSharedImmutable, "go/bin/compile", backend.Unbounded); !errors.Is(err, most.ErrIntegrityViolation) {
		t.Fatalf("tampered NVMe shared block err = %v, want ErrIntegrityViolation", err)
	}
	_ = keyA
	_ = keyB
}

func TestTenantRevocationAndConcurrency(t *testing.T) {
	ctx := context.Background()
	mem := storetest.NewMem()
	nvmeDir := t.TempDir()

	eng, err := most.NewEngine(mem, most.Config{
		Policy:         most.PolicySelectiveMirror,
		NVMeDir:        nvmeDir,
		TotalRAMBytes:  32 << 20,
		TotalNVMeBytes: 128 << 20,
		HotThreshold:   2.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	cA := newTestCipher(t, "a-key")
	cB := newTestCipher(t, "b-key")
	_ = eng.RegisterTenant("a", "vol-a", cA, 8<<20, 32<<20)
	_ = eng.RegisterTenant("b", "vol-b", cB, 8<<20, 32<<20)

	for i := range 8 {
		_, _ = eng.PublishShared(ctx, fmt.Sprintf("tc-%d", i), testPayload(byte(i+1), 4096))
		_ = eng.Put(ctx, "a", most.ScopePrivate, fmt.Sprintf("pa-%d", i), testPayload(byte(i+10), 4096))
		_ = eng.Put(ctx, "b", most.ScopePrivate, fmt.Sprintf("pb-%d", i), testPayload(byte(i+20), 4096))
	}

	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range 32 {
				k := fmt.Sprintf("tc-%d", i%8)
				if _, err := eng.Get(ctx, "a", most.ScopeSharedImmutable, k, backend.Unbounded); err != nil {
					t.Errorf("tenant a shared get: %v", err)
				}
				pk := fmt.Sprintf("pa-%d", i%8)
				if _, err := eng.Get(ctx, "a", most.ScopePrivate, pk, backend.Unbounded); err != nil {
					t.Errorf("tenant a private get: %v", err)
				}
				if id%2 == 0 {
					_, _ = eng.Get(ctx, "b", most.ScopePrivate, fmt.Sprintf("pb-%d", i%8), backend.Unbounded)
				}
			}
		}(g)
	}
	wg.Wait()

	if eng.ResidentRAMBytes("b") == 0 || eng.ResidentNVMeBytes("b") == 0 {
		t.Fatalf("expected tenant b to hold resident RAM/NVMe before revoke: ram=%d nvme=%d",
			eng.ResidentRAMBytes("b"), eng.ResidentNVMeBytes("b"))
	}

	if err := eng.RevokeTenant("b"); err != nil {
		t.Fatal(err)
	}
	stB := eng.TenantStats("b")
	if !stB.Revoked || stB.ResidentRAMBytes != 0 || stB.ResidentNVMeBytes != 0 {
		t.Fatalf("revoked tenant b still holds cache: %+v", stB)
	}
	if _, err := eng.Get(ctx, "b", most.ScopePrivate, "pb-0", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("revoked tenant b Get err = %v, want ErrTenantUnauthorized", err)
	}
	if _, err := eng.Get(ctx, "b", most.ScopeSharedImmutable, "tc-0", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("revoked tenant b shared Get err = %v, want ErrTenantUnauthorized", err)
	}

	// Healthy tenant A remains intact and serves from RAM.
	if eng.ResidentRAMBytes("a") == 0 {
		t.Fatalf("healthy tenant a lost RAM residency")
	}
	got, err := eng.Get(ctx, "a", most.ScopePrivate, "pa-0", backend.Unbounded)
	if err != nil || !bytes.Equal(got, testPayload(10, 4096)) {
		t.Fatalf("healthy tenant a read failed: %v", err)
	}
	_ = blake3.Size
}

func TestTenantBackendContractAndConcurrency(t *testing.T) {
	ctx := context.Background()
	openTB := func(t *testing.T) backend.Backend {
		t.Helper()
		mem := storetest.NewMem()
		eng, err := most.NewEngine(mem, most.Config{
			Policy:         most.PolicySelectiveMirror,
			TotalRAMBytes:  32 << 20,
			TotalNVMeBytes: 128 << 20,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = eng.Close() })
		if err := eng.RegisterTenant("conf", "vol-conf", newTestCipher(t, "conf-key"), 16<<20, 64<<20); err != nil {
			t.Fatal(err)
		}
		return most.NewTenantBackend(eng, "conf")
	}

	t.Run("get put range delete versioned", func(t *testing.T) {
		be := openTB(t)
		if _, _, err := be.Get(ctx, "a/b", backend.Unbounded); !errors.Is(err, backend.ErrNotFound) {
			t.Fatalf("Get absent: %v", err)
		}
		if err := be.Put(ctx, "a/b", []byte("0123456789")); err != nil {
			t.Fatal(err)
		}
		got, v, err := be.Get(ctx, "a/b", backend.Unbounded)
		if err != nil || string(got) != "0123456789" || v == "" {
			t.Fatalf("Get: %q %q %v", got, v, err)
		}
		if _, v2, err := be.Get(ctx, "a/b", backend.Unbounded); err != nil || v2 != v {
			t.Fatalf("version moved: %q -> %q (%v)", v, v2, err)
		}
		r, err := be.GetRange(ctx, "a/b", 3, 4)
		if err != nil || string(r) != "3456" {
			t.Fatalf("GetRange: %q %v", r, err)
		}
		if _, err := be.GetRange(ctx, "a/b", 8, 5); err == nil {
			t.Fatal("expected out-of-range error")
		}
		if n, err := be.Stat(ctx, "a/b"); err != nil || n != 10 {
			t.Fatalf("Stat = %d (%v), want 10", n, err)
		}
		v3, err := be.PutIfVersion(ctx, "a/b", []byte("updated"), v)
		if err != nil || v3 == v {
			t.Fatalf("PutIfVersion = %q (%v)", v3, err)
		}
		if _, err := be.PutIfVersion(ctx, "a/b", []byte("stale"), v); !errors.Is(err, backend.ErrConflict) {
			t.Fatalf("stale PutIfVersion err = %v, want ErrConflict", err)
		}
		if got2, _, err := be.Get(ctx, "a/b", backend.Unbounded); err != nil || string(got2) != "updated" {
			t.Fatalf("after PutIfVersion got %q (%v)", got2, err)
		}
		if err := be.Delete(ctx, "a/b"); err != nil {
			t.Fatal(err)
		}
		if _, _, err := be.Get(ctx, "a/b", backend.Unbounded); !errors.Is(err, backend.ErrNotFound) {
			t.Fatalf("Get after Delete: %v", err)
		}
	})

	t.Run("cas race and large object", func(t *testing.T) {
		be := openTB(t)
		if _, err := be.PutIfAbsent(ctx, "head", []byte("base")); err != nil {
			t.Fatal(err)
		}
		if _, err := be.PutIfAbsent(ctx, "head", []byte("dup")); !errors.Is(err, backend.ErrConflict) {
			t.Fatalf("PutIfAbsent dup err = %v", err)
		}
		for round := range 10 {
			_, v, err := be.Get(ctx, "head", backend.Unbounded)
			if err != nil {
				t.Fatal(err)
			}
			var wins int32
			var mu sync.Mutex
			var wg sync.WaitGroup
			start := make(chan struct{})
			for i := range 8 {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					<-start
					if _, err := be.PutIfVersion(ctx, "head", fmt.Appendf(nil, "r%d-w%d", round, id), v); err == nil {
						mu.Lock()
						wins++
						mu.Unlock()
					} else if !errors.Is(err, backend.ErrConflict) {
						t.Errorf("unexpected CAS error: %v", err)
					}
				}(i)
			}
			close(start)
			wg.Wait()
			if wins != 1 {
				t.Fatalf("round %d wins = %d, want 1", round, wins)
			}
		}

		big := bytes.Repeat([]byte("0123456789abcdef"), 1<<18) // 4 MiB
		if err := be.Put(ctx, "pack/aa/big", big); err != nil {
			t.Fatal(err)
		}
		got, _, err := be.Get(ctx, "pack/aa/big", backend.Unbounded)
		if err != nil || !bytes.Equal(got, big) {
			t.Fatalf("large object Get failed: %v", err)
		}
		seg, err := be.GetRange(ctx, "pack/aa/big", 1<<20+5, 1000)
		if err != nil || !bytes.Equal(seg, big[1<<20+5:1<<20+5+1000]) {
			t.Fatalf("large object GetRange failed: %v", err)
		}
	})

	t.Run("list prefix and max bound", func(t *testing.T) {
		be := openTB(t)
		for _, k := range []string{"pack/aa/1", "pack/aa/2", "pack/bb/3", "head"} {
			if err := be.Put(ctx, k, []byte("payload-"+k)); err != nil {
				t.Fatal(err)
			}
		}
		ch, errCh := be.List(ctx, "pack/aa/")
		var keys []string
		for o := range ch {
			keys = append(keys, o.Key)
		}
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
		if len(keys) != 2 {
			t.Fatalf("List(pack/aa/) = %v, want 2 keys", keys)
		}
		if _, _, err := be.Get(ctx, "pack/aa/1", 4); !errors.Is(err, backend.ErrTooLarge) {
			t.Fatalf("bounded Get err = %v, want ErrTooLarge", err)
		}
	})
}

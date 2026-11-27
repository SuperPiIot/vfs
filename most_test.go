// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package vfs_test

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/hanzoai/vfs"
	"github.com/hanzoai/vfs/internal/storetest"
	"github.com/hanzoai/vfs/pkg/backend"
	"github.com/hanzoai/vfs/pkg/backend/s3"
	"github.com/hanzoai/vfs/pkg/most"
)

func openS3At(t testing.TB, srv *storetest.S3, prefix string) backend.Backend {
	t.Helper()
	be, err := s3.New(s3.Config{
		Endpoint:  srv.Endpoint(),
		Insecure:  true,
		Region:    "us-east-1",
		Bucket:    srv.Bucket,
		Prefix:    prefix,
		AccessKey: "most-access",
		SecretKey: "most-secret",
	})
	if err != nil {
		t.Fatal(err)
	}
	return be
}

// TestMOSTSharedImmutableToolchainWithRealVolume mounts an immutable shared
// base toolchain Volume and multiple concurrent per-agent private Volumes
// backed by most.TenantBackend over a live storetest.S3 server.
func TestMOSTSharedImmutableToolchainWithRealVolume(t *testing.T) {
	srv := storetest.NewS3(t, "most-bucket")
	sharedBE := openS3At(t, srv, "toolchain")
	tcCipher := vfs.Hybrid(newKey(t))

	if err := vfs.Create(ctx, cfg(sharedBE, tcCipher, 0)); err != nil {
		t.Fatal(err)
	}
	baseVol, err := vfs.Attach(ctx, cfg(sharedBE, tcCipher, 1))
	if err != nil {
		t.Fatal(err)
	}

	usr := mkdir(t, baseVol, vfs.RootInode, "usr")
	lib := mkdir(t, baseVol, usr, "lib")
	tcDir := mkdir(t, baseVol, lib, "toolchain")

	type tcFile struct {
		path string
		data []byte
	}
	var toolchainFiles []tcFile
	// Small files (inside meta/ shards) + chunked files (inside pack/ objects).
	for i := range 8 {
		nm := fmt.Sprintf("hdr-%02d.h", i)
		payload := content(uint64(100+i), 2048+i*128)
		write(t, baseVol, tcDir, nm, payload)
		toolchainFiles = append(toolchainFiles, tcFile{path: "/usr/lib/toolchain/" + nm, data: payload})
	}
	for i := range 2 {
		nm := fmt.Sprintf("libtc-%02d.a", i)
		payload := content(uint64(500+i), 24<<10)
		write(t, baseVol, tcDir, nm, payload)
		toolchainFiles = append(toolchainFiles, tcFile{path: "/usr/lib/toolchain/" + nm, data: payload})
	}
	flush(t, baseVol)
	if err := baseVol.Close(); err != nil {
		t.Fatal(err)
	}

	// Measure uncached baseline: 4 cold agents opening sharedBE and reading every toolchain file.
	reqsBeforeBaseline := srv.Requests()
	for range 4 {
		ro, err := vfs.Open(ctx, vfs.Config{Backend: sharedBE, Cipher: tcCipher, Cache: 128 << 10})
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range toolchainFiles {
			in, err := ro.Resolve(ctx, f.path)
			if err != nil {
				t.Fatal(err)
			}
			got := readAll(t, ro, in.ID)
			mustEqual(t, "baseline "+f.path, got, f.data)
		}
		_ = ro.Close()
	}
	baselineS3Reqs := srv.Requests() - reqsBeforeBaseline

	// Now mount 4 concurrent agents via most.Engine over root S3 bucket.
	rootS3 := openS3At(t, srv, "")
	nvmeDir := t.TempDir()
	eng, err := most.NewEngine(rootS3, most.Config{
		Policy:         most.PolicySelectiveMirror,
		NVMeDir:        nvmeDir,
		TotalRAMBytes:  32 << 20,
		TotalNVMeBytes: 128 << 20,
		HotThreshold:   2.0,
		SharedPrefix:   "toolchain",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	const agents = 4
	agentCiphers := make([]vfs.Cipher, agents)
	for i := range agents {
		agentID := fmt.Sprintf("agent-%d", i)
		agentCiphers[i] = vfs.Hybrid(newKey(t))
		if err := eng.RegisterTenant(agentID, fmt.Sprintf("vol-%d", i), agentCiphers[i], 8<<20, 32<<20); err != nil {
			t.Fatal(err)
		}
	}

	// Each agent attaches its own private writable Volume and writes private secrets.
	secretToken := "confidential-agent-secret-payload-9911"
	secretFile := "private-secret-credentials.env"
	for i := range agents {
		agentID := fmt.Sprintf("agent-%d", i)
		privBE := most.NewTenantBackend(eng, agentID)
		if err := vfs.Create(ctx, cfg(privBE, agentCiphers[i], 0)); err != nil {
			t.Fatal(err)
		}
		av, err := vfs.Attach(ctx, cfg(privBE, agentCiphers[i], 1))
		if err != nil {
			t.Fatal(err)
		}
		ws := mkdir(t, av, vfs.RootInode, "workspace")
		write(t, av, ws, secretFile, []byte(fmt.Sprintf("%s-%d", secretToken, i)))
		flush(t, av)
		if err := av.Close(); err != nil {
			t.Fatal(err)
		}

		// Reopen agent private Volume via most.NewTenantBackend and verify exact private read.
		roPriv, err := vfs.Open(ctx, cfg(most.NewTenantBackend(eng, agentID), agentCiphers[i], 0))
		if err != nil {
			t.Fatal(err)
		}
		in, err := roPriv.Resolve(ctx, "/workspace/"+secretFile)
		if err != nil {
			t.Fatal(err)
		}
		mustEqual(t, agentID+" private file", readAll(t, roPriv, in.ID), []byte(fmt.Sprintf("%s-%d", secretToken, i)))
		_ = roPriv.Close()
	}

	// Rule 1 check: verify zero plaintext payload or filename leakage in S3 objects and NVMe disk.
	ch, errCh := rootS3.List(ctx, "")
	for o := range ch {
		if bytes.Contains([]byte(o.Key), []byte(secretFile)) {
			t.Fatalf("S3 key %q leaked plaintext filename", o.Key)
		}
		raw, _, gerr := rootS3.Get(ctx, o.Key, backend.Unbounded)
		if gerr == nil && (bytes.Contains(raw, []byte(secretToken)) || bytes.Contains(raw, []byte(secretFile))) {
			t.Fatalf("S3 object %q leaked plaintext content/filename", o.Key)
		}
	}
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	_ = filepath.WalkDir(nvmeDir, func(path string, d os.DirEntry, werr error) error {
		if werr != nil || d.IsDir() {
			return nil
		}
		raw, _ := os.ReadFile(path)
		if bytes.Contains(raw, []byte(secretToken)) || bytes.Contains(raw, []byte(secretFile)) {
			t.Fatalf("NVMe cache file %s leaked plaintext", path)
		}
		return nil
	})

	// Shared toolchain warm/concurrent read test across 4 agents (2 cold-volume passes each = 8 mounts).
	eng.ResetStats()
	reqsBeforeMOST := srv.Requests()
	for pass := range 2 {
		var wg sync.WaitGroup
		for i := range agents {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				agentID := fmt.Sprintf("agent-%d", id)
				ro, err := vfs.Open(ctx, vfs.Config{
					Backend: most.NewSharedBackend(eng, agentID),
					Cipher:  tcCipher,
					Cache:   128 << 10,
				})
				if err != nil {
					t.Errorf("agent %d open shared: %v", id, err)
					return
				}
				defer ro.Close()
				for _, f := range toolchainFiles {
					in, err := ro.Resolve(ctx, f.path)
					if err != nil {
						t.Errorf("agent %d resolve %s: %v", id, f.path, err)
						return
					}
					got := readAll(t, ro, in.ID)
					if !bytes.Equal(got, f.data) {
						t.Errorf("pass %d agent %d file %s mismatch", pass, id, f.path)
						return
					}
				}
			}(i)
		}
		wg.Wait()
	}
	mostS3Reqs := srv.Requests() - reqsBeforeMOST
	st := eng.Stats()

	// Compare per-4-agent-pass S3 requests: MOST shared across 8 mounts must issue far fewer
	// S3 requests than 2x the 4-agent uncached baseline (2 * baselineS3Reqs), with 0 demotion writes.
	if mostS3Reqs >= 2*baselineS3Reqs {
		t.Fatalf("MOST S3 requests (%d over 8 mounts) did not reduce vs uncached baseline (%d over 4 mounts)", mostS3Reqs, baselineS3Reqs)
	}
	if st.RAMHits == 0 || st.NVMeHits == 0 || st.MirrorPromotions == 0 {
		t.Fatalf("expected RAMHits, NVMeHits, and MirrorPromotions > 0, got %+v", st)
	}
	if st.TierMigrationWrites != 0 {
		t.Fatalf("MOST incurred TierMigrationWrites=%d, want 0", st.TierMigrationWrites)
	}
	t.Logf("S3 requests: uncached 4-agent=%d (8-agent est=%d), MOST 8-agent=%d; RAMHits=%d NVMeHits=%d MirrorPromotions=%d BackendGets=%d P99=%v",
		baselineS3Reqs, 2*baselineS3Reqs, mostS3Reqs, st.RAMHits, st.NVMeHits, st.MirrorPromotions, st.BackendGets, st.P99)
}

// TestMOSTTenantIsolationPressureAndRevocation verifies per-tenant RAM/NVMe quota
// isolation against a noisy-neighbor cold-scan flood, fail-closed cross-tenant
// ciphertext swap detection with vfs.Hybrid, and immediate zeroing on RevokeTenant.
func TestMOSTTenantIsolationPressureAndRevocation(t *testing.T) {
	mem := storetest.NewMem()
	nvmeDir := t.TempDir()
	eng, err := most.NewEngine(mem, most.Config{
		Policy:         most.PolicySelectiveMirror,
		NVMeDir:        nvmeDir,
		TotalRAMBytes:  128 << 10,
		TotalNVMeBytes: 512 << 10,
		HotThreshold:   2.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	// Share the SAME vfs.Hybrid cipher intentionally to prove AAD [tenantID|volID|key] stops swaps!
	sharedHybrid := vfs.Hybrid(newKey(t))
	const (
		blockSize  = 4096
		quotaRAM   = 64 << 10
		quotaNVMe  = 256 << 10
		hotCount   = 8
		floodCount = 120
	)
	if err := eng.RegisterTenant("tenant-a", "vol-a", sharedHybrid, quotaRAM, quotaNVMe); err != nil {
		t.Fatal(err)
	}
	if err := eng.RegisterTenant("tenant-b", "vol-b", sharedHybrid, quotaRAM, quotaNVMe); err != nil {
		t.Fatal(err)
	}

	// Warm Tenant A's hot working set (8 x 4 KiB = 32 KiB) into RAM.
	for i := range hotCount {
		k := fmt.Sprintf("hot-a-%02d", i)
		if err := eng.Put(ctx, "tenant-a", most.ScopePrivate, k, content(uint64(100+i), blockSize)); err != nil {
			t.Fatal(err)
		}
		for range 3 {
			if _, err := eng.Get(ctx, "tenant-a", most.ScopePrivate, k, backend.Unbounded); err != nil {
				t.Fatal(err)
			}
		}
	}
	beforeRAM := eng.ResidentRAMBytes("tenant-a")
	if beforeRAM != hotCount*blockSize {
		t.Fatalf("tenant-a ResidentRAMBytes = %d, want %d", beforeRAM, hotCount*blockSize)
	}

	// Noisy-neighbor Tenant B floods 120 cold 4 KiB private keys (plus 120 cold shared keys).
	for i := range floodCount {
		kb := fmt.Sprintf("flood-b-%03d", i)
		if err := eng.Put(ctx, "tenant-b", most.ScopePrivate, kb, content(uint64(2000+i), blockSize)); err != nil {
			t.Fatal(err)
		}
		if _, err := eng.Get(ctx, "tenant-b", most.ScopePrivate, kb, backend.Unbounded); err != nil {
			t.Fatal(err)
		}
		ks := fmt.Sprintf("shared-cold-%03d", i)
		if _, err := eng.PublishShared(ctx, ks, content(uint64(5000+i), blockSize)); err != nil {
			t.Fatal(err)
		}
		if _, err := eng.Get(ctx, "tenant-b", most.ScopeSharedImmutable, ks, backend.Unbounded); err != nil {
			t.Fatal(err)
		}
	}

	// Tenant A MUST keep 100% of its resident RAM and hit RAM on every subsequent read.
	if got := eng.ResidentRAMBytes("tenant-a"); got != beforeRAM {
		t.Fatalf("noisy neighbor evicted tenant-a RAM: got %d, want %d", got, beforeRAM)
	}
	if got := eng.ResidentNVMeBytes("tenant-b"); got > quotaNVMe {
		t.Fatalf("tenant-b exceeded NVMe quota: got %d > %d", got, quotaNVMe)
	}
	stABefore := eng.TenantStats("tenant-a")
	for i := range hotCount {
		k := fmt.Sprintf("hot-a-%02d", i)
		got, err := eng.Get(ctx, "tenant-a", most.ScopePrivate, k, backend.Unbounded)
		if err != nil {
			t.Fatal(err)
		}
		mustEqual(t, k, got, content(uint64(100+i), blockSize))
	}
	stAAfter := eng.TenantStats("tenant-a")
	if stAAfter.RAMHits-stABefore.RAMHits != hotCount || stAAfter.BackendGets != stABefore.BackendGets {
		t.Fatalf("tenant-a lost RAM hits during flood: before=%+v after=%+v", stABefore, stAAfter)
	}

	// Ciphertext swap across tenants (even sharing vfs.Hybrid!) fails closed with ErrIntegrityViolation.
	swapMem := storetest.NewMem()
	engSwap, err := most.NewEngine(swapMem, most.Config{Policy: most.PolicySelectiveMirror})
	if err != nil {
		t.Fatal(err)
	}
	defer engSwap.Close()
	_ = engSwap.RegisterTenant("a", "v1", sharedHybrid, 64<<10, 64<<10)
	_ = engSwap.RegisterTenant("b", "v2", sharedHybrid, 64<<10, 64<<10)
	_ = engSwap.Put(ctx, "a", most.ScopePrivate, "secret", []byte("alpha"))
	keyA := swapMem.Keys()[0]
	sealedA, _, _ := swapMem.Get(ctx, keyA, backend.Unbounded)
	_ = engSwap.Put(ctx, "b", most.ScopePrivate, "secret", []byte("beta"))
	var keyB string
	for _, k := range swapMem.Keys() {
		if k != keyA {
			keyB = k
		}
	}
	_ = swapMem.Put(ctx, keyB, sealedA)
	engCold, err := most.NewEngine(swapMem, most.Config{Policy: most.PolicySelectiveMirror})
	if err != nil {
		t.Fatal(err)
	}
	defer engCold.Close()
	_ = engCold.RegisterTenant("b", "v2", sharedHybrid, 64<<10, 64<<10)
	if _, err := engCold.Get(ctx, "b", most.ScopePrivate, "secret", backend.Unbounded); !errors.Is(err, most.ErrIntegrityViolation) {
		t.Fatalf("cross-tenant Hybrid ciphertext swap err = %v, want ErrIntegrityViolation", err)
	}

	// RevokeTenant immediately zeros/purges Tenant B RAM and NVMe files without hurting Tenant A.
	if err := eng.RevokeTenant("tenant-b"); err != nil {
		t.Fatal(err)
	}
	stB := eng.TenantStats("tenant-b")
	if !stB.Revoked || stB.ResidentRAMBytes != 0 || stB.ResidentNVMeBytes != 0 {
		t.Fatalf("revoked tenant-b stats not zeroed: %+v", stB)
	}
	if _, err := eng.Get(ctx, "tenant-b", most.ScopePrivate, "flood-b-000", backend.Unbounded); !errors.Is(err, most.ErrTenantUnauthorized) {
		t.Fatalf("revoked tenant-b Get err = %v, want ErrTenantUnauthorized", err)
	}
	if eng.ResidentRAMBytes("tenant-a") != beforeRAM {
		t.Fatalf("RevokeTenant(b) disturbed healthy tenant-a RAM: %d vs %d", eng.ResidentRAMBytes("tenant-a"), beforeRAM)
	}
	if _, err := eng.Get(ctx, "tenant-a", most.ScopePrivate, "hot-a-00", backend.Unbounded); err != nil {
		t.Fatalf("healthy tenant-a read after RevokeTenant(b): %v", err)
	}
	t.Logf("Tenant A P99=%v ResidentRAM=%d; Tenant B post-revoke RAM=%d NVMe=%d",
		stAAfter.P99, eng.ResidentRAMBytes("tenant-a"), stB.ResidentRAMBytes, stB.ResidentNVMeBytes)
}

// TestMOSTSelectiveMirrorVsTierMigrationZipf runs an identical Zipfian shared
// toolchain + cold-scan workload across PolicySelectiveMirror and PolicyTierMigration
// using disk-backed NVMe and compares real p99, demotion writes, and cache hits.
func TestMOSTSelectiveMirrorVsTierMigrationZipf(t *testing.T) {
	const (
		numSharedHot   = 48
		numColdScan    = 144
		blockSize      = 8192
		ramCapacity    = 16 * blockSize  // ~33% of active shared toolchain fits in RAM
		nvmeCapacity   = 128 * blockSize // NVMe holds active toolchain + scan window
		opsPerAgent    = 400
		numAgents      = 4
		zipfExponent   = 1.15
		coldScanPeriod = 4 // every 4th op is a 1-hit cold scan
	)

	type runOut struct {
		stats most.Stats
		mem   int64
	}

	runPolicy := func(policy most.Policy) runOut {
		nvmeDir := t.TempDir()
		be := storetest.NewMem()
		eng, err := most.NewEngine(be, most.Config{
			Policy:         policy,
			NVMeDir:        nvmeDir,
			TotalRAMBytes:  ramCapacity,
			TotalNVMeBytes: nvmeCapacity,
			HotThreshold:   2.0,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eng.Close()

		ciph := newTestCipherForZipf(t)
		for i := range numAgents {
			if err := eng.RegisterTenant(fmt.Sprintf("ag-%d", i), fmt.Sprintf("v-%d", i), ciph, ramCapacity, nvmeCapacity); err != nil {
				t.Fatal(err)
			}
		}

		for i := range numSharedHot {
			if _, err := eng.PublishShared(ctx, fmt.Sprintf("tc-%03d", i), content(uint64(100+i), blockSize)); err != nil {
				t.Fatal(err)
			}
		}
		for i := range numColdScan {
			if _, err := eng.PublishShared(ctx, fmt.Sprintf("scan-%04d", i), content(uint64(10000+i), blockSize)); err != nil {
				t.Fatal(err)
			}
		}

		// Precompute deterministic Zipfian weights.
		cdf := make([]float64, numSharedHot)
		var totalW float64
		for i := range numSharedHot {
			w := 1.0 / math.Pow(float64(i+1), zipfExponent)
			totalW += w
			cdf[i] = totalW
		}
		sampleZipf := func(rng *rand.Rand) int {
			u := rng.Float64() * totalW
			lo, hi := 0, numSharedHot-1
			for lo < hi {
				mid := (lo + hi) / 2
				if cdf[mid] < u {
					lo = mid + 1
				} else {
					hi = mid
				}
			}
			return lo
		}

		var wg sync.WaitGroup
		for a := range numAgents {
			wg.Add(1)
			go func(agentIdx int) {
				defer wg.Done()
				rng := rand.New(rand.NewPCG(uint64(42+agentIdx), uint64(99+agentIdx)))
				tid := fmt.Sprintf("ag-%d", agentIdx)
				scanBase := agentIdx * (numColdScan / numAgents)
				scanCursor := 0
				for step := range opsPerAgent {
					var key string
					if step%coldScanPeriod == 0 && scanCursor < (numColdScan/numAgents) {
						key = fmt.Sprintf("scan-%04d", scanBase+scanCursor)
						scanCursor++
					} else {
						key = fmt.Sprintf("tc-%03d", sampleZipf(rng))
					}
					if _, err := eng.Get(ctx, tid, most.ScopeSharedImmutable, key, backend.Unbounded); err != nil {
						t.Errorf("%s Get(%s): %v", tid, key, err)
						return
					}
				}
			}(a)
		}
		wg.Wait()

		st := eng.Stats()
		var ramTotal int64
		for i := range numAgents {
			ramTotal += eng.ResidentRAMBytes(fmt.Sprintf("ag-%d", i))
		}
		return runOut{stats: st, mem: ramTotal}
	}

	mostRes := runPolicy(most.PolicySelectiveMirror)
	migRes := runPolicy(most.PolicyTierMigration)

	if mostRes.stats.TierMigrationWrites != 0 {
		t.Fatalf("MOST TierMigrationWrites = %d, want 0", mostRes.stats.TierMigrationWrites)
	}
	if migRes.stats.TierMigrationWrites == 0 {
		t.Fatalf("PolicyTierMigration TierMigrationWrites = 0, expected > 0")
	}
	if mostRes.stats.RAMHits <= migRes.stats.RAMHits {
		t.Fatalf("expected MOST RAMHits (%d) > TierMigration RAMHits (%d)", mostRes.stats.RAMHits, migRes.stats.RAMHits)
	}
	t.Logf("MOST      : P50=%v P95=%v P99=%v BackendGets=%d RAMHits=%d NVMeHits=%d Promotions=%d DemotionWrites=%d GlobalRAM=%d",
		mostRes.stats.P50, mostRes.stats.P95, mostRes.stats.P99,
		mostRes.stats.BackendGets, mostRes.stats.RAMHits, mostRes.stats.NVMeHits,
		mostRes.stats.MirrorPromotions, mostRes.stats.TierMigrationWrites, mostRes.stats.ResidentRAMBytes)
	t.Logf("TierMigrat: P50=%v P95=%v P99=%v BackendGets=%d RAMHits=%d NVMeHits=%d DemotionWrites=%d GlobalRAM=%d",
		migRes.stats.P50, migRes.stats.P95, migRes.stats.P99,
		migRes.stats.BackendGets, migRes.stats.RAMHits, migRes.stats.NVMeHits,
		migRes.stats.TierMigrationWrites, migRes.stats.ResidentRAMBytes)
	_ = time.Now()
}

func newTestCipherForZipf(t testing.TB) vfs.Cipher {
	t.Helper()
	return vfs.Hybrid(newKey(t))
}

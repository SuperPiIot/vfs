// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Package most implements Mirror-Optimized Storage Tiering (MOST, FAST '26)
// for hanzoai/vfs: selective hot-data mirroring between a disk-backed NVMe
// cache tier and a tenant-bounded host RAM pool over an authenticated object
// backend, alongside an ordinary exclusive cache-tier migration baseline.
package most

import (
	"bytes"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/hanzoai/vfs/internal/blake3"
)

// Policy selects the caching and promotion behavior of an Engine.
type Policy uint8

const (
	// PolicySelectiveMirror implements MOST: shared immutable and cold blocks
	// stay in the NVMe tier by default, one-hit scans never pollute RAM, and
	// only keys whose hotness score reaches HotThreshold are mirrored into RAM
	// while keeping the NVMe copy intact (zero demotion write-back on eviction).
	PolicySelectiveMirror Policy = iota

	// PolicyTierMigration implements ordinary exclusive cache-tier migration:
	// cold misses enter RAM, RAM evictions demote/write back to NVMe, and NVMe
	// hits migrate to RAM by removing the NVMe copy.
	PolicyTierMigration
)

func (p Policy) String() string {
	switch p {
	case PolicySelectiveMirror:
		return "selective_mirror"
	case PolicyTierMigration:
		return "tier_migration"
	default:
		return fmt.Sprintf("policy(%d)", p)
	}
}

// Scope partitions objects between tenant-private encrypted volumes and
// content-verified read-only shared immutable toolchains.
type Scope uint8

const (
	// ScopePrivate designates tenant-owned objects sealed with the tenant's
	// Cipher and bound via AAD to [tenantID | volID | key].
	ScopePrivate Scope = iota

	// ScopeSharedImmutable designates read-only toolchain objects verified by
	// BLAKE3 content digest and shared only among authorized tenants.
	ScopeSharedImmutable
)

func (s Scope) String() string {
	switch s {
	case ScopePrivate:
		return "private"
	case ScopeSharedImmutable:
		return "shared_immutable"
	default:
		return fmt.Sprintf("scope(%d)", s)
	}
}

// Cipher matches vfs.Cipher without introducing a circular import on package vfs.
// Implementations must authenticate as well as encrypt and be concurrency-safe.
type Cipher interface {
	Seal(plaintext []byte) (sealed []byte, keyID string, err error)
	Open(sealed []byte, keyID string) (plaintext []byte, err error)
}

var (
	// ErrTenantUnauthorized is returned fail-closed when a caller is unknown,
	// revoked, lacks a registered Cipher for private operations, or fails a
	// shared-toolchain ACL check.
	ErrTenantUnauthorized = errors.New("most: tenant unauthorized")

	// ErrIntegrityViolation is returned fail-closed when authenticated AAD
	// verification, Cipher tag checking, or BLAKE3 content digest checking fails.
	ErrIntegrityViolation = errors.New("most: integrity violation")

	// ErrReadOnly is returned when attempting to mutate an immutable shared tier.
	ErrReadOnly = errors.New("most: shared immutable tier is read-only")

	// ErrInvalidConfig is returned when Engine or tenant parameters are invalid.
	ErrInvalidConfig = errors.New("most: invalid configuration")
)

const (
	// UnlimitedQuota lets a tenant share up to the Engine's global tier limit.
	UnlimitedQuota int64 = -1

	defaultTotalRAMBytes  int64         = 64 << 20
	defaultTotalNVMeBytes int64         = 512 << 20
	defaultHotThreshold   float64       = 2.0
	defaultDecayHalfLife  time.Duration = 30 * time.Second
	defaultMaxHotEntries  int           = 262144

	privateFrameMagic = "MOST\x01"
	sharedFrameMagic  = "MIMG\x01"
	aadLen            = blake3.Size
)

// Config configures a MOST Engine instance.
type Config struct {
	Policy         Policy
	NVMeDir        string
	TotalRAMBytes  int64
	TotalNVMeBytes int64
	HotThreshold   float64
	DecayHalfLife  time.Duration
	WindowOps      uint64
	MaxHotEntries  int
	SharedPrefix   string
	SyncNVMe       bool
	Clock          func() time.Time
}

func (c Config) withDefaults() (Config, error) {
	if c.Policy != PolicySelectiveMirror && c.Policy != PolicyTierMigration {
		return c, fmt.Errorf("%w: unknown policy %d", ErrInvalidConfig, c.Policy)
	}
	if c.TotalRAMBytes == 0 {
		c.TotalRAMBytes = defaultTotalRAMBytes
	}
	if c.TotalRAMBytes < 0 {
		return c, fmt.Errorf("%w: TotalRAMBytes must be >= 0", ErrInvalidConfig)
	}
	if c.TotalNVMeBytes == 0 {
		c.TotalNVMeBytes = defaultTotalNVMeBytes
	}
	if c.TotalNVMeBytes < 0 {
		return c, fmt.Errorf("%w: TotalNVMeBytes must be >= 0", ErrInvalidConfig)
	}
	if c.HotThreshold <= 0 {
		c.HotThreshold = defaultHotThreshold
	}
	if c.DecayHalfLife == 0 {
		c.DecayHalfLife = defaultDecayHalfLife
	}
	if c.MaxHotEntries <= 0 {
		c.MaxHotEntries = defaultMaxHotEntries
	}
	if c.Clock == nil {
		c.Clock = time.Now
	}
	return c, nil
}

// Stats captures Engine-wide cache, write-amplification, security, and
// exact nanosecond read-latency percentiles.
type Stats struct {
	Policy               Policy
	TotalReads           int64
	BackendGets          int64
	RAMHits              int64
	NVMeHits             int64
	MirrorPromotions     int64
	TierMigrationWrites  int64
	NVMeWrites           int64
	RAMEvictions         int64
	NVMeEvictions        int64
	ResidentRAMBytes     int64
	ResidentNVMeBytes    int64
	IntegrityViolations  int64
	UnauthorizedAttempts int64
	P50                  time.Duration
	P95                  time.Duration
	P99                  time.Duration
}

// TenantStats captures per-tenant residency, quota bounds, cache efficiency,
// and exact read-latency percentiles.
type TenantStats struct {
	TenantID            string
	VolID               string
	Registered          bool
	Revoked             bool
	SharedAllowed       bool
	QuotaRAMBytes       int64
	QuotaNVMeBytes      int64
	ResidentRAMBytes    int64
	ResidentNVMeBytes   int64
	TotalReads          int64
	BackendGets         int64
	RAMHits             int64
	NVMeHits            int64
	MirrorPromotions    int64
	TierMigrationWrites int64
	NVMeWrites          int64
	P50                 time.Duration
	P95                 time.Duration
	P99                 time.Duration
}

type latencyRecorder struct {
	mu      sync.Mutex
	samples []int64
}

func (l *latencyRecorder) record(d time.Duration) {
	ns := d.Nanoseconds()
	l.mu.Lock()
	l.samples = append(l.samples, ns)
	l.mu.Unlock()
}

func (l *latencyRecorder) reset() {
	l.mu.Lock()
	l.samples = l.samples[:0]
	l.mu.Unlock()
}

func (l *latencyRecorder) snapshot() []int64 {
	l.mu.Lock()
	cp := append([]int64(nil), l.samples...)
	l.mu.Unlock()
	return cp
}

func (l *latencyRecorder) percentiles() (p50, p95, p99 time.Duration) {
	return computePercentiles(l.snapshot())
}

func computePercentiles(samples []int64) (p50, p95, p99 time.Duration) {
	if len(samples) == 0 {
		return 0, 0, 0
	}
	s := slices.Clone(samples)
	slices.Sort(s)
	at := func(q float64) time.Duration {
		idx := min(len(s)-1, max(0, int(math.Ceil(q*float64(len(s))))-1))
		return time.Duration(s[idx]) * time.Nanosecond
	}
	return at(0.50), at(0.95), at(0.99)
}

type hotEntry struct {
	score    float64
	lastTime time.Time
	lastOp   uint64
	lastSeq  uint64
}

// hotTracker tracks sliding-window / exponentially decayed hit frequencies.
// Decay uses a short grace window (halfLife/64) so rapid consecutive hits
// accumulate cleanly (1st=1.0, 2nd=2.0 >= HotThreshold=2.0) while stale
// singletons decay smoothly over DecayHalfLife.
type hotTracker struct {
	halfLife  time.Duration
	grace     time.Duration
	windowOps uint64
	maxItems  int
	opSeq     uint64
	entries   map[string]hotEntry
}

func newHotTracker(halfLife time.Duration, windowOps uint64, maxItems int) *hotTracker {
	var grace time.Duration
	if halfLife > 0 {
		grace = halfLife / 64
	}
	return &hotTracker{
		halfLife:  halfLife,
		grace:     grace,
		windowOps: windowOps,
		maxItems:  maxItems,
		entries:   make(map[string]hotEntry),
	}
}

func (h *hotTracker) decayedScore(e hotEntry, now time.Time) float64 {
	score := e.score
	if score <= 0 {
		return 0
	}
	if h.halfLife > 0 {
		dt := now.Sub(e.lastTime)
		if dt > h.grace {
			score *= math.Pow(0.5, float64(dt)/float64(h.halfLife))
		}
	}
	if h.windowOps > 0 && h.opSeq > e.lastOp {
		dops := h.opSeq - e.lastOp
		if dops > h.windowOps/64 {
			score *= math.Pow(0.5, float64(dops)/float64(h.windowOps))
		}
	}
	return score
}

func (h *hotTracker) touch(key string, now time.Time) float64 {
	h.opSeq++
	cur, ok := h.entries[key]
	if !ok {
		if len(h.entries) >= h.maxItems {
			h.evictColdest(now)
		}
		h.entries[key] = hotEntry{score: 1.0, lastTime: now, lastOp: h.opSeq, lastSeq: h.opSeq}
		return 1.0
	}
	score := h.decayedScore(cur, now) + 1.0
	h.entries[key] = hotEntry{score: score, lastTime: now, lastOp: h.opSeq, lastSeq: h.opSeq}
	return score
}

func (h *hotTracker) score(key string, now time.Time) float64 {
	cur, ok := h.entries[key]
	if !ok {
		return 0
	}
	return h.decayedScore(cur, now)
}

func (h *hotTracker) forget(key string) {
	delete(h.entries, key)
}

func (h *hotTracker) evictColdest(now time.Time) {
	// Sample a small bounded batch rather than scanning the full map so O(1) cost holds.
	const sampleBudget = 32
	var victim string
	minScore := math.Inf(1)
	checked := 0
	for k, v := range h.entries {
		s := h.decayedScore(v, now)
		if s < minScore {
			minScore = s
			victim = k
		}
		checked++
		if checked >= sampleBudget {
			break
		}
	}
	if victim != "" {
		delete(h.entries, victim)
	}
}

func bindContext(kind string, fields ...string) [aadLen]byte {
	h := blake3.New()
	_, _ = h.WriteString("hanzoai/vfs/most context")
	writeField := func(s string) {
		var lenBuf [binary.MaxVarintLen64]byte
		n := binary.PutUvarint(lenBuf[:], uint64(len(s)))
		_, _ = h.Write(lenBuf[:n])
		_, _ = h.WriteString(s)
	}
	writeField(kind)
	for _, f := range fields {
		writeField(f)
	}
	var out [aadLen]byte
	h.Sum(out[:0])
	return out
}

func privateAAD(tenantID, volID, key string) [aadLen]byte {
	return bindContext("private", tenantID, volID, key)
}

func sealPrivate(c Cipher, tenantID, volID, key string, plain []byte) ([]byte, error) {
	if c == nil {
		return nil, ErrTenantUnauthorized
	}
	aad := privateAAD(tenantID, volID, key)
	buf := make([]byte, 0, aadLen+len(plain))
	buf = append(buf, aad[:]...)
	buf = append(buf, plain...)
	sealed, keyID, err := c.Seal(buf)
	clear(buf)
	if err != nil {
		return nil, fmt.Errorf("most: seal failed: %w", err)
	}
	if bytes.Contains(sealed, aad[:]) {
		return nil, fmt.Errorf("%w: cipher output leaks authenticated context", ErrIntegrityViolation)
	}
	if len(keyID) == 0 || len(keyID) > 255 {
		return nil, fmt.Errorf("%w: invalid keyID length %d", ErrIntegrityViolation, len(keyID))
	}
	out := make([]byte, 0, len(privateFrameMagic)+1+len(keyID)+len(sealed))
	out = append(out, privateFrameMagic...)
	out = append(out, byte(len(keyID)))
	out = append(out, keyID...)
	out = append(out, sealed...)
	return out, nil
}

func unsealPrivate(c Cipher, tenantID, volID, key string, frame []byte) ([]byte, error) {
	if c == nil {
		return nil, ErrTenantUnauthorized
	}
	hdr := len(privateFrameMagic) + 1
	if len(frame) < hdr || string(frame[:len(privateFrameMagic)]) != privateFrameMagic {
		return nil, fmt.Errorf("%w: invalid sealed frame header", ErrIntegrityViolation)
	}
	keyLen := int(frame[len(privateFrameMagic)])
	if keyLen == 0 || len(frame) < hdr+keyLen {
		return nil, fmt.Errorf("%w: truncated keyID in sealed frame", ErrIntegrityViolation)
	}
	keyID := string(frame[hdr : hdr+keyLen])
	ciphertext := frame[hdr+keyLen:]
	dec, err := c.Open(ciphertext, keyID)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrIntegrityViolation, err)
	}
	if len(dec) < aadLen {
		clear(dec)
		return nil, fmt.Errorf("%w: missing authenticated AAD header", ErrIntegrityViolation)
	}
	aad := privateAAD(tenantID, volID, key)
	if subtle.ConstantTimeCompare(dec[:aadLen], aad[:]) != 1 {
		clear(dec)
		return nil, fmt.Errorf("%w: tenant/volume/key AAD mismatch", ErrIntegrityViolation)
	}
	plain := append([]byte(nil), dec[aadLen:]...)
	clear(dec)
	return plain, nil
}

func encodeSharedFrame(digest [blake3.Size]byte, plain []byte) []byte {
	out := make([]byte, 0, len(sharedFrameMagic)+blake3.Size+len(plain))
	out = append(out, sharedFrameMagic...)
	out = append(out, digest[:]...)
	out = append(out, plain...)
	return out
}

func decodeSharedFrame(frame []byte, expected [blake3.Size]byte) ([]byte, error) {
	hdr := len(sharedFrameMagic) + blake3.Size
	if len(frame) < hdr || string(frame[:len(sharedFrameMagic)]) != sharedFrameMagic {
		return nil, fmt.Errorf("%w: invalid shared frame header", ErrIntegrityViolation)
	}
	var stored [blake3.Size]byte
	copy(stored[:], frame[len(sharedFrameMagic):hdr])
	payload := frame[hdr:]
	sum := blake3.Sum(payload)
	if subtle.ConstantTimeCompare(sum[:], stored[:]) != 1 {
		return nil, fmt.Errorf("%w: shared block checksum mismatch", ErrIntegrityViolation)
	}
	if expected != ([blake3.Size]byte{}) && subtle.ConstantTimeCompare(sum[:], expected[:]) != 1 {
		return nil, fmt.Errorf("%w: shared block does not match pinned digest", ErrIntegrityViolation)
	}
	return append([]byte(nil), payload...), nil
}

func hashHex(kind string, parts ...string) string {
	d := bindContext(kind, parts...)
	return hex.EncodeToString(d[:])
}

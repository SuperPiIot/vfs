// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package dogi

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"sort"
	"sync"

	"github.com/hanzoai/vfs/internal/blake3"
	"github.com/hanzoai/vfs/pkg/backend"
	"github.com/hanzoai/vfs/pkg/fence"
)

const (
	// DefaultPackTarget is the standard open pack flush threshold (16 MiB),
	// matching vfs v2 pack sizing while staying well below the 64 MiB single-PUT limit.
	DefaultPackTarget = 16 << 20

	// DefaultMaxManifestBytes bounds the fenced "head" object, matching vfs/head.go.
	DefaultMaxManifestBytes = 64 << 10

	// maxPackObjectBytes bounds a single sealed pack read/write under S3's 64 MiB atomic PUT ceiling.
	maxPackObjectBytes = 60 << 20

	// maxManifestReadBytes bounds a sealed meta/ checkpoint manifest read.
	maxManifestReadBytes = 32 << 20

	headKey       = "head"
	bindLen       = 32
	headMagic     = "DGHD\x01"
	manifestMagic = "DGMN\x01"
)

var (
	// ErrBadCipher is returned when a Cipher leaks plaintext, fails round-trip, or accepts tampered bytes.
	ErrBadCipher = errors.New("dogi: cipher failed validation")

	// ErrCorrupt is returned when a sealed head, manifest, or chunk fails authentication or integrity verification.
	ErrCorrupt = errors.New("dogi: corrupt or mismatched object")

	// ErrPackConflict is returned when an immutable pack/<2hex>/<blake3> key exists with conflicting size or bytes.
	ErrPackConflict = errors.New("dogi: immutable pack key conflict with differing payload")

	// ErrReadOnly is returned on mutating operations against a read-only Packer.
	ErrReadOnly = errors.New("dogi: packer is read-only")
)

// Cipher seals and opens every chunk, manifest, and checkpoint head written to
// the backend. Implementations must authenticate ciphertext and be safe for
// concurrent use. Compatible with vfs.Cipher (including vfs.Hybrid).
type Cipher interface {
	Seal(plaintext []byte) (sealed []byte, keyID string, err error)
	Open(sealed []byte, keyID string) (plaintext []byte, err error)
}

// Mode controls how many open pack streams the unified Packer maintains.
type Mode uint8

const (
	// ModeBaselineSingleStream uses NumStreams=1 (naive FIFO co-packing).
	ModeBaselineSingleStream Mode = iota
	// ModeLifetimeSeparated uses NumStreams=3 (one open pack stream per Lifetime class).
	ModeLifetimeSeparated
)

// String returns a human-readable mode identifier.
func (m Mode) String() string {
	switch m {
	case ModeBaselineSingleStream:
		return "baseline-fifo-1stream"
	case ModeLifetimeSeparated:
		return "dogi-lifetime-3stream"
	default:
		return fmt.Sprintf("mode(%d)", uint8(m))
	}
}

// NumStreams returns the number of active open pack buffers for m.
func (m Mode) NumStreams() int {
	if m == ModeLifetimeSeparated {
		return NumLifetimes
	}
	return 1
}

// Config configures a unified DOGI Packer instance.
type Config struct {
	Backend                  backend.Backend
	Fence                    *fence.Store // optional; created with MaxManifestBytes if nil
	Cipher                   Cipher
	Round                    uint64    // fencing round (> 0 and < math.MaxUint64 for writers)
	VolumeID                 [16]byte  // optional; generated randomly on fresh creation if zero
	Mode                     Mode      // ModeBaselineSingleStream or ModeLifetimeSeparated
	PackTarget               int       // open pack flush threshold in bytes (defaults to 16 MiB)
	Predictor                Predictor // optional classifier/predictor for ModeLifetimeSeparated
	MaxManifestBytes         int64     // fenced "head" object ceiling (defaults to 65536)
	ReadOnly                 bool      // opens existing checkpoint without claiming or writing
	DisableSurvivorPromotion bool      // if false, GC promotes surviving ephemeral chunks to longer-lived streams
}

// Stats captures exact byte, pack, and recovery accounting for a Packer.
type Stats struct {
	LogicalChunksWritten   int64 `json:"logical_chunks_written"`
	LogicalPlaintextBytes  int64 `json:"logical_plaintext_bytes"`
	LogicalSealedBytes     int64 `json:"logical_sealed_bytes"`
	PhysicalBytesWritten   int64 `json:"physical_bytes_written"`
	PhysicalBytesReclaimed int64 `json:"physical_bytes_reclaimed"`
	LiveBytesCopied        int64 `json:"live_bytes_copied"`
	PacksCreated           int64 `json:"packs_created"`
	PacksDeleted           int64 `json:"packs_deleted"`
	LostACKRecoveries      int64 `json:"lost_ack_recoveries"`
	PackDedupReuses        int64 `json:"pack_dedup_reuses"`
	Rollbacks              int64 `json:"rollbacks"`
}

// ActivePhysicalBytes returns the total sealed pack bytes currently stored on the backend.
func (s Stats) ActivePhysicalBytes() int64 {
	return s.PhysicalBytesWritten - s.PhysicalBytesReclaimed
}

// WriteAmplification returns PhysicalBytesWritten / LogicalSealedBytes (1.0 when zero extra copying occurred).
func (s Stats) WriteAmplification() float64 {
	if s.LogicalSealedBytes <= 0 {
		return 1.0
	}
	return float64(s.PhysicalBytesWritten) / float64(s.LogicalSealedBytes)
}

// PackRecord describes a single immutable pack/<2hex>/<blake3> object.
type PackRecord struct {
	Seq             uint64
	Key             string
	BLAKE3          [32]byte
	StreamIdx       int
	PredictedClass  Lifetime
	SealedBytes     int64
	LiveSealedBytes int64
	DeadSealedBytes int64
	ChunkCount      int
	LiveChunkCount  int
	Deleted         bool
	Committed       bool
}

// ChunkRecord locates one live object chunk either inside an active open stream
// buffer (PackSeq == 0) or inside a flushed pack object (PackSeq > 0).
type ChunkRecord struct {
	LogicalKey string
	PlainHash  [32]byte
	PlainLen   uint64
	PackSeq    uint64
	StreamIdx  int
	Off        uint64
	SealedLen  uint64
	Class      Lifetime
	Version    uint64
}

type bufferedChunk struct {
	LogicalKey string
	PlainHash  [32]byte
	PlainLen   uint64
	Frame      []byte
	Class      Lifetime
	Version    uint64
}

type streamBuffer struct {
	items       []bufferedChunk
	totalSealed int64
}

type packPin struct {
	readers  int
	draining bool
}

type packerSnapshot struct {
	seq          uint64
	nextPackSeq  uint64
	nextChunkSeq uint64
	streams      []streamBuffer
	packs        map[uint64]PackRecord
	packOrder    []uint64
	packsByKey   map[string]int
	live         map[string]ChunkRecord
	stats        Stats
}

// Packer is the unified canonical object packer supporting both 1-stream FIFO
// baseline and 3-stream lifetime-separated DOGI placement on the same engine.
type Packer struct {
	cfg      Config
	be       backend.Backend
	fence    *fence.Store
	cipher   Cipher
	volID    [16]byte
	round    uint64
	readOnly bool

	wmu     sync.Mutex // serializes mutations, flushes, checkpoints, and compaction
	mu      sync.Mutex // protects in-memory state and reader pins; never held across backend I/O
	pinCond *sync.Cond

	fenced       bool
	seq          uint64
	nextPackSeq  uint64
	nextChunkSeq uint64
	streams      []streamBuffer
	packs        map[uint64]PackRecord
	packOrder    []uint64
	packsByKey   map[string]int
	pins         map[uint64]*packPin
	live         map[string]ChunkRecord
	stats        Stats
}

func validateConfig(cfg Config) (Config, error) {
	if cfg.Backend == nil || cfg.Cipher == nil {
		return cfg, errors.New("dogi: config requires Backend and Cipher")
	}
	if cfg.PackTarget <= 0 {
		cfg.PackTarget = DefaultPackTarget
	}
	if cfg.PackTarget > 32<<20 {
		return cfg, fmt.Errorf("dogi: PackTarget %d exceeds 32 MiB ceiling", cfg.PackTarget)
	}
	if cfg.MaxManifestBytes <= 0 {
		cfg.MaxManifestBytes = DefaultMaxManifestBytes
	}
	if !cfg.ReadOnly && (cfg.Round == 0 || cfg.Round == math.MaxUint64) {
		return cfg, fmt.Errorf("%w: writer requires Round in [1, 2^64-2], got %d", fence.ErrStaleRound, cfg.Round)
	}
	if err := checkCipher(cfg.Cipher); err != nil {
		return cfg, err
	}
	if cfg.Fence == nil {
		cfg.Fence = fence.New(cfg.Backend, cfg.MaxManifestBytes)
	}
	return cfg, nil
}

// NewPacker creates a fresh Packer in memory (claiming nothing until Checkpoint,
// or reading an existing checkpoint if AttachPacker is used).
func NewPacker(ctx context.Context, cfg Config) (*Packer, error) {
	if ctx == nil {
		return nil, errors.New("dogi: nil context")
	}
	var err error
	cfg, err = validateConfig(cfg)
	if err != nil {
		return nil, err
	}
	volID := cfg.VolumeID
	if volID == ([16]byte{}) {
		if _, err := rand.Read(volID[:]); err != nil {
			return nil, fmt.Errorf("dogi: generate volume id: %w", err)
		}
	}
	return newEmptyPacker(cfg, volID), nil
}

// OpenPacker opens the latest committed checkpoint on cfg.Backend in read-only mode.
func OpenPacker(ctx context.Context, cfg Config) (*Packer, error) {
	if ctx == nil {
		return nil, errors.New("dogi: nil context")
	}
	cfg.ReadOnly = true
	var err error
	cfg, err = validateConfig(cfg)
	if err != nil {
		return nil, err
	}
	payload, _, err := cfg.Fence.Get(ctx, headKey)
	if err != nil {
		return nil, err
	}
	hd, err := decodeHead(cfg.Cipher, cfg.Backend.String(), payload)
	if err != nil {
		return nil, err
	}
	man, err := fetchManifest(ctx, cfg.Backend, cfg.Cipher, hd)
	if err != nil {
		return nil, err
	}
	p := newEmptyPacker(cfg, hd.VolumeID)
	p.hydrateFromManifest(hd, man)
	return p, nil
}

// AttachPacker takes fenced ownership of cfg.Backend at cfg.Round. If a "head"
// checkpoint exists, it performs an atomic fence.Claim takeover (rejecting
// cfg.Round <= recorded with fence.ErrStaleRound) and hydrates the latest
// committed manifest; if no "head" exists yet, it initializes a fresh Packer.
func AttachPacker(ctx context.Context, cfg Config) (*Packer, error) {
	if ctx == nil {
		return nil, errors.New("dogi: nil context")
	}
	cfg.ReadOnly = false
	var err error
	cfg, err = validateConfig(cfg)
	if err != nil {
		return nil, err
	}
	var (
		hasHead bool
		hd      checkpointHead
	)
	err = cfg.Fence.Claim(ctx, headKey, cfg.Round, func(payload []byte, recorded uint64) error {
		if payload == nil {
			hasHead = false
			return nil
		}
		if cfg.Round <= recorded {
			return fmt.Errorf("%w: attach round %d <= recorded %d", fence.ErrStaleRound, cfg.Round, recorded)
		}
		decoded, derr := decodeHead(cfg.Cipher, cfg.Backend.String(), payload)
		if derr != nil {
			return derr
		}
		hasHead = true
		hd = decoded
		return nil
	})
	if err != nil {
		return nil, err
	}
	if !hasHead {
		volID := cfg.VolumeID
		if volID == ([16]byte{}) {
			if _, err := rand.Read(volID[:]); err != nil {
				return nil, fmt.Errorf("dogi: generate volume id: %w", err)
			}
		}
		return newEmptyPacker(cfg, volID), nil
	}
	man, err := fetchManifest(ctx, cfg.Backend, cfg.Cipher, hd)
	if err != nil {
		return nil, err
	}
	p := newEmptyPacker(cfg, hd.VolumeID)
	p.hydrateFromManifest(hd, man)
	return p, nil
}

func newEmptyPacker(cfg Config, volID [16]byte) *Packer {
	nStreams := cfg.Mode.NumStreams()
	p := &Packer{
		cfg:        cfg,
		be:         cfg.Backend,
		fence:      cfg.Fence,
		cipher:     cfg.Cipher,
		volID:      volID,
		round:      cfg.Round,
		readOnly:   cfg.ReadOnly,
		streams:    make([]streamBuffer, nStreams),
		packs:      make(map[uint64]PackRecord),
		packsByKey: make(map[string]int),
		pins:       make(map[uint64]*packPin),
		live:       make(map[string]ChunkRecord),
	}
	p.pinCond = sync.NewCond(&p.mu)
	return p
}

func (p *Packer) hydrateFromManifest(hd checkpointHead, man checkpointManifest) {
	p.seq = hd.Seq
	p.nextPackSeq = man.NextPackSeq
	p.nextChunkSeq = man.NextChunkSeq
	p.stats = man.Stats
	p.packs = make(map[uint64]PackRecord, len(man.Packs))
	p.packOrder = make([]uint64, 0, len(man.Packs))
	p.packsByKey = make(map[string]int)
	p.pins = make(map[uint64]*packPin, len(man.Packs))
	for _, pk := range man.Packs {
		pk.Committed = true
		p.packs[pk.Seq] = pk
		p.packOrder = append(p.packOrder, pk.Seq)
		if !pk.Deleted {
			p.packsByKey[pk.Key]++
			p.pins[pk.Seq] = &packPin{}
		}
	}
	p.live = make(map[string]ChunkRecord, len(man.Live))
	for _, c := range man.Live {
		p.live[c.LogicalKey] = c
	}
}

// VolumeID returns the 16-byte sealing domain identifier of p.
func (p *Packer) VolumeID() [16]byte { return p.volID }

// Round returns the fencing round held by p.
func (p *Packer) Round() uint64 { return p.round }

// Seq returns the latest committed checkpoint sequence number.
func (p *Packer) Seq() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.seq
}

// Stats returns a snapshot of p's exact byte and pack accounting counters.
func (p *Packer) Stats() Stats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

// Packs returns a copy of all pack records in creation sequence order.
func (p *Packer) Packs() []PackRecord {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]PackRecord, 0, len(p.packOrder))
	for _, seq := range p.packOrder {
		if pk, ok := p.packs[seq]; ok {
			out = append(out, pk)
		}
	}
	return out
}

// LiveKeys returns sorted logical keys currently live in p.
func (p *Packer) LiveKeys() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	keys := make([]string, 0, len(p.live))
	for k := range p.live {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Put seals data and places it into the open pack stream selected by p.cfg.Mode
// (stream 0 in baseline mode; predicted Lifetime stream in DOGI mode). If the
// target stream reaches PackTarget, it flushes that pack to the backend via
// immutable PutIfAbsent; on upload failure, stream and accounting state are
// atomically rolled back.
func (p *Packer) Put(ctx context.Context, logicalKey string, data []byte, meta ObjectMeta) error {
	if ctx == nil {
		return errors.New("dogi: nil context")
	}
	if logicalKey == "" {
		return errors.New("dogi: empty logical key")
	}
	if p.readOnly {
		return ErrReadOnly
	}
	if meta.Path == "" {
		meta.Path = logicalKey
	}
	if meta.Size == 0 {
		meta.Size = int64(len(data))
	}

	plainHash := blake3.Sum(data)
	frame, err := sealFrame(p.cipher, chunkAAD(p.volID, plainHash), data)
	if err != nil {
		return err
	}

	streamIdx, class := p.route(meta)

	p.wmu.Lock()
	defer p.wmu.Unlock()

	p.mu.Lock()
	if p.fenced {
		p.mu.Unlock()
		return fence.ErrStaleRound
	}
	snap := p.snapshotLocked()

	p.removeChunkLocked(logicalKey)
	p.nextChunkSeq++
	ver := p.nextChunkSeq
	bc := bufferedChunk{
		LogicalKey: logicalKey,
		PlainHash:  plainHash,
		PlainLen:   uint64(len(data)),
		Frame:      frame,
		Class:      class,
		Version:    ver,
	}
	st := &p.streams[streamIdx]
	off := uint64(st.totalSealed)
	st.items = append(st.items, bc)
	st.totalSealed += int64(len(frame))
	p.live[logicalKey] = ChunkRecord{
		LogicalKey: logicalKey,
		PlainHash:  plainHash,
		PlainLen:   uint64(len(data)),
		PackSeq:    0,
		StreamIdx:  streamIdx,
		Off:        off,
		SealedLen:  uint64(len(frame)),
		Class:      class,
		Version:    ver,
	}
	p.stats.LogicalChunksWritten++
	p.stats.LogicalPlaintextBytes += int64(len(data))
	p.stats.LogicalSealedBytes += int64(len(frame))

	needFlush := st.totalSealed >= int64(p.cfg.PackTarget)
	var staged stagedPack
	if needFlush {
		staged = p.stageStreamLocked(streamIdx)
	}
	p.mu.Unlock()

	if !needFlush {
		return nil
	}

	res, err := p.uploadStagedPack(ctx, staged)
	if err != nil {
		p.mu.Lock()
		p.restoreLocked(snap)
		p.stats.Rollbacks++
		p.mu.Unlock()
		return err
	}

	p.mu.Lock()
	p.commitStagedPackLocked(staged, res, false)
	p.mu.Unlock()
	return nil
}

// Delete removes logicalKey from the live index, immediately crediting dead
// bytes on its backing pack (or removing it from the open stream buffer if
// not yet flushed). Deleting a missing key is a no-op.
func (p *Packer) Delete(logicalKey string) error {
	if p.readOnly {
		return ErrReadOnly
	}
	p.wmu.Lock()
	defer p.wmu.Unlock()

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.fenced {
		return fence.ErrStaleRound
	}
	p.removeChunkLocked(logicalKey)
	return nil
}

// Get fetches, decrypts, and verifies the plaintext of logicalKey. Concurrent
// Get calls safely pin backing packs against deletion during Compact relocation
// without holding p.mu across backend reads.
func (p *Packer) Get(ctx context.Context, logicalKey string) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("dogi: nil context")
	}

	p.mu.Lock()
	c, ok := p.live[logicalKey]
	if !ok {
		p.mu.Unlock()
		return nil, fmt.Errorf("%w: %s", fs.ErrNotExist, logicalKey)
	}
	if c.PackSeq == 0 {
		var frame []byte
		st := p.streams[c.StreamIdx]
		for _, item := range st.items {
			if item.LogicalKey == logicalKey && item.Version == c.Version {
				frame = append([]byte(nil), item.Frame...)
				break
			}
		}
		p.mu.Unlock()
		if frame == nil {
			return nil, fmt.Errorf("%w: buffered chunk missing for %s", ErrCorrupt, logicalKey)
		}
		return p.openVerifiedChunk(c, frame)
	}

	pk, ok := p.packs[c.PackSeq]
	pin := p.pins[c.PackSeq]
	if !ok || pk.Deleted || pin == nil {
		p.mu.Unlock()
		return nil, fmt.Errorf("%w: backing pack %d missing for %s", ErrCorrupt, c.PackSeq, logicalKey)
	}
	pin.readers++
	p.mu.Unlock()

	defer func() {
		p.mu.Lock()
		pin.readers--
		if pin.readers == 0 && pin.draining {
			p.pinCond.Broadcast()
		}
		p.mu.Unlock()
	}()

	frame, err := p.be.GetRange(ctx, pk.Key, int64(c.Off), int64(c.SealedLen))
	if err != nil {
		return nil, fmt.Errorf("dogi: read chunk %s from %s: %w", logicalKey, pk.Key, err)
	}
	return p.openVerifiedChunk(c, frame)
}

func (p *Packer) openVerifiedChunk(c ChunkRecord, frame []byte) ([]byte, error) {
	plain, err := unsealFrame(p.cipher, chunkAAD(p.volID, c.PlainHash), frame)
	if err != nil {
		return nil, fmt.Errorf("dogi: unseal %s: %w", c.LogicalKey, err)
	}
	if uint64(len(plain)) != c.PlainLen || blake3.Sum(plain) != c.PlainHash {
		return nil, fmt.Errorf("%w: chunk digest/length mismatch for %s", ErrCorrupt, c.LogicalKey)
	}
	return plain, nil
}

// FlushStreams uploads all non-empty open stream buffers as immutable packs
// without updating "head". On any upload failure, state rolls back atomically.
func (p *Packer) FlushStreams(ctx context.Context) error {
	if ctx == nil {
		return errors.New("dogi: nil context")
	}
	if p.readOnly {
		return ErrReadOnly
	}
	p.wmu.Lock()
	defer p.wmu.Unlock()
	return p.flushStreamsLocked(ctx)
}

func (p *Packer) flushStreamsLocked(ctx context.Context) error {
	p.mu.Lock()
	if p.fenced {
		p.mu.Unlock()
		return fence.ErrStaleRound
	}
	snap := p.snapshotLocked()
	var stagedList []stagedPack
	for idx := range p.streams {
		if len(p.streams[idx].items) > 0 {
			stagedList = append(stagedList, p.stageStreamLocked(idx))
		}
	}
	p.mu.Unlock()

	if len(stagedList) == 0 {
		return nil
	}

	results := make([]uploadedPackResult, len(stagedList))
	for i, st := range stagedList {
		res, err := p.uploadStagedPack(ctx, st)
		if err != nil {
			p.mu.Lock()
			p.restoreLocked(snap)
			p.stats.Rollbacks++
			p.mu.Unlock()
			return err
		}
		results[i] = res
	}

	p.mu.Lock()
	for i, st := range stagedList {
		p.commitStagedPackLocked(st, results[i], false)
	}
	p.mu.Unlock()
	return nil
}

// Flush flushes all open pack streams and commits an atomic fenced checkpoint
// on "head".
func (p *Packer) Flush(ctx context.Context) error {
	return p.Checkpoint(ctx)
}

// Checkpoint flushes any buffered chunks to immutable pack/<2hex>/<blake3>
// objects, seals and writes the checkpoint manifest to meta/<2hex>/<blake3>,
// and atomically advances "head" via pkg/fence.Store. If any upload or fenced
// CAS fails (and did not land), pending state rolls back atomically so
// CheckAccounting stays exact.
func (p *Packer) Checkpoint(ctx context.Context) error {
	if ctx == nil {
		return errors.New("dogi: nil context")
	}
	if p.readOnly {
		return ErrReadOnly
	}
	p.wmu.Lock()
	defer p.wmu.Unlock()
	return p.checkpointWithWriterLock(ctx)
}

func (p *Packer) checkpointWithWriterLock(ctx context.Context) error {
	p.mu.Lock()
	if p.fenced {
		p.mu.Unlock()
		return fence.ErrStaleRound
	}
	snap := p.snapshotLocked()
	var stagedList []stagedPack
	for idx := range p.streams {
		if len(p.streams[idx].items) > 0 {
			stagedList = append(stagedList, p.stageStreamLocked(idx))
		}
	}
	p.mu.Unlock()

	results := make([]uploadedPackResult, len(stagedList))
	for i, st := range stagedList {
		res, err := p.uploadStagedPack(ctx, st)
		if err != nil {
			p.mu.Lock()
			p.restoreLocked(snap)
			p.stats.Rollbacks++
			p.mu.Unlock()
			return err
		}
		results[i] = res
	}

	p.mu.Lock()
	for i, st := range stagedList {
		p.commitStagedPackLocked(st, results[i], true)
	}
	nextSeq := p.seq + 1
	man := p.buildManifestLocked(nextSeq)
	p.mu.Unlock()

	if err := p.commitManifestAndHead(ctx, nextSeq, man); err != nil {
		p.mu.Lock()
		p.restoreLocked(snap)
		if errors.Is(err, fence.ErrStaleRound) {
			p.fenced = true
		} else {
			p.stats.Rollbacks++
		}
		p.mu.Unlock()
		return err
	}

	p.mu.Lock()
	p.seq = nextSeq
	for _, seq := range p.packOrder {
		if pk, ok := p.packs[seq]; ok && !pk.Committed {
			pk.Committed = true
			p.packs[seq] = pk
		}
	}
	p.mu.Unlock()
	return nil
}

func (p *Packer) commitManifestAndHead(ctx context.Context, seq uint64, man checkpointManifest) error {
	rawMan := man.marshal()
	sealedMan, err := sealFrame(p.cipher, manifestAAD(p.volID, seq), rawMan)
	if err != nil {
		return err
	}
	manHash := blake3.Sum(sealedMan)
	manKey := prefixedHashKey("meta", manHash)
	if _, recovered, err := p.putImmutable(ctx, manKey, sealedMan); err != nil {
		return fmt.Errorf("dogi: put manifest %s: %w", manKey, err)
	} else if recovered {
		p.mu.Lock()
		p.stats.LostACKRecoveries++
		p.mu.Unlock()
	}

	hd := checkpointHead{
		VolumeID:    p.volID,
		Seq:         seq,
		Mode:        p.cfg.Mode,
		PackTarget:  uint64(p.cfg.PackTarget),
		ManifestRef: manHash,
		Stats:       man.Stats,
	}
	sealedHead, err := sealFrame(p.cipher, headAAD(p.be.String()), hd.marshal())
	if err != nil {
		return err
	}
	if err := p.fence.Put(ctx, headKey, sealedHead, p.round); err != nil {
		// Check whether the head write landed despite a lost ACK or concurrent takeover.
		if p.checkHeadLanded(ctx, hd) {
			p.mu.Lock()
			p.stats.LostACKRecoveries++
			p.mu.Unlock()
			if errors.Is(err, fence.ErrStaleRound) {
				p.mu.Lock()
				p.fenced = true
				p.mu.Unlock()
			}
			return nil
		}
		return err
	}
	return nil
}

func (p *Packer) checkHeadLanded(ctx context.Context, want checkpointHead) bool {
	payload, _, err := p.fence.Get(ctx, headKey)
	if err != nil {
		return false
	}
	got, err := decodeHead(p.cipher, p.be.String(), payload)
	if err != nil {
		return false
	}
	return got.VolumeID == want.VolumeID && got.Seq == want.Seq && got.ManifestRef == want.ManifestRef
}

func (p *Packer) route(meta ObjectMeta) (streamIdx int, class Lifetime) {
	if p.cfg.Mode == ModeBaselineSingleStream {
		return 0, LifetimeMedium
	}
	class = LifetimeMedium
	if p.cfg.Predictor != nil {
		class = p.cfg.Predictor.Classify(meta)
	}
	if !class.Valid() {
		class = LifetimeMedium
	}
	return int(class), class
}

func (p *Packer) removeChunkLocked(logicalKey string) {
	old, ok := p.live[logicalKey]
	if !ok {
		return
	}
	delete(p.live, logicalKey)
	if old.PackSeq == 0 {
		st := &p.streams[old.StreamIdx]
		filtered := st.items[:0]
		var total int64
		for _, item := range st.items {
			if item.LogicalKey == logicalKey && item.Version == old.Version {
				continue
			}
			filtered = append(filtered, item)
			total += int64(len(item.Frame))
		}
		st.items = filtered
		st.totalSealed = total
		// Refresh stream offsets for remaining buffered chunks so Get offsets stay exact.
		var off uint64
		for _, item := range st.items {
			if cur, exists := p.live[item.LogicalKey]; exists && cur.PackSeq == 0 && cur.Version == item.Version {
				cur.Off = off
				p.live[item.LogicalKey] = cur
			}
			off += uint64(len(item.Frame))
		}
		return
	}
	pk, exists := p.packs[old.PackSeq]
	if !exists || pk.Deleted {
		return
	}
	pk.LiveSealedBytes -= int64(old.SealedLen)
	pk.DeadSealedBytes += int64(old.SealedLen)
	pk.LiveChunkCount--
	p.packs[old.PackSeq] = pk
}

type stagedPack struct {
	seq       uint64
	streamIdx int
	class     Lifetime
	items     []bufferedChunk
	payload   []byte
	hash      [32]byte
	key       string
}

type uploadedPackResult struct {
	deduped   bool
	recovered bool
}

func (p *Packer) stageStreamLocked(streamIdx int) stagedPack {
	st := &p.streams[streamIdx]
	p.nextPackSeq++
	seq := p.nextPackSeq
	items := make([]bufferedChunk, len(st.items))
	copy(items, st.items)
	payload := make([]byte, 0, st.totalSealed)
	for _, it := range items {
		payload = append(payload, it.Frame...)
	}
	st.items = nil
	st.totalSealed = 0

	hash := blake3.Sum(payload)
	key := prefixedHashKey("pack", hash)

	class := LifetimeMedium
	if p.cfg.Mode == ModeLifetimeSeparated && streamIdx >= 0 && streamIdx < NumLifetimes {
		class = Lifetime(streamIdx)
	}
	return stagedPack{
		seq:       seq,
		streamIdx: streamIdx,
		class:     class,
		items:     items,
		payload:   payload,
		hash:      hash,
		key:       key,
	}
}

func (p *Packer) uploadStagedPack(ctx context.Context, st stagedPack) (uploadedPackResult, error) {
	deduped, recovered, err := p.putImmutable(ctx, st.key, st.payload)
	if err != nil {
		return uploadedPackResult{}, err
	}
	return uploadedPackResult{deduped: deduped, recovered: recovered}, nil
}

func (p *Packer) commitStagedPackLocked(st stagedPack, res uploadedPackResult, committed bool) {
	alreadyInStore := p.packsByKey[st.key] > 0
	p.packsByKey[st.key]++
	p.pins[st.seq] = &packPin{}

	var off uint64
	var liveBytes int64
	var liveCount int
	for _, it := range st.items {
		n := uint64(len(it.Frame))
		if cur, ok := p.live[it.LogicalKey]; ok && cur.PackSeq == 0 && cur.Version == it.Version {
			cur.PackSeq = st.seq
			cur.Off = off
			p.live[it.LogicalKey] = cur
			liveBytes += int64(n)
			liveCount++
		}
		off += n
	}
	sealedTotal := int64(len(st.payload))
	p.packs[st.seq] = PackRecord{
		Seq:             st.seq,
		Key:             st.key,
		BLAKE3:          st.hash,
		StreamIdx:       st.streamIdx,
		PredictedClass:  st.class,
		SealedBytes:     sealedTotal,
		LiveSealedBytes: liveBytes,
		DeadSealedBytes: sealedTotal - liveBytes,
		ChunkCount:      len(st.items),
		LiveChunkCount:  liveCount,
		Committed:       committed,
	}
	p.packOrder = append(p.packOrder, st.seq)
	p.stats.PacksCreated++
	if res.recovered {
		p.stats.LostACKRecoveries++
	}
	if res.deduped && alreadyInStore {
		p.stats.PackDedupReuses++
	} else {
		p.stats.PhysicalBytesWritten += sealedTotal
	}
}

// putImmutable writes sealed data once under a content-addressed key via
// PutIfAbsent. On ErrConflict it checks the existing object's size and exact
// bytes: identical bytes reuse/dedup safely, whereas differing size or bytes
// fail closed with ErrPackConflict (never overwriting). On non-conflict upload
// errors (such as a lost ACK after write landed), it verifies whether the exact
// bytes are present on the backend and recovers cleanly if so.
func (p *Packer) putImmutable(ctx context.Context, key string, data []byte) (deduped bool, recovered bool, err error) {
	_, err = p.be.PutIfAbsent(ctx, key, data)
	if err == nil {
		return false, false, nil
	}
	if errors.Is(err, backend.ErrConflict) {
		if verr := p.verifyExistingImmutable(ctx, key, data); verr != nil {
			return false, false, verr
		}
		return true, false, nil
	}
	// Check whether the write actually landed before an acknowledgment failure.
	if verr := p.verifyExistingImmutable(ctx, key, data); verr == nil {
		return false, true, nil
	}
	return false, false, err
}

func (p *Packer) verifyExistingImmutable(ctx context.Context, key string, want []byte) error {
	sz, err := p.be.Stat(ctx, key)
	if err != nil {
		return err
	}
	if sz != int64(len(want)) {
		return fmt.Errorf("%w: %s stored size %d != payload size %d", ErrPackConflict, key, sz, len(want))
	}
	existing, _, err := p.be.Get(ctx, key, int64(len(want))+1)
	if err != nil {
		return err
	}
	if len(existing) != len(want) || !bytes.Equal(existing, want) {
		return fmt.Errorf("%w: %s existing payload differs from %x", ErrPackConflict, key, blake3.Sum(want))
	}
	return nil
}

func (p *Packer) snapshotLocked() packerSnapshot {
	streams := make([]streamBuffer, len(p.streams))
	for i, st := range p.streams {
		cp := make([]bufferedChunk, len(st.items))
		copy(cp, st.items)
		streams[i] = streamBuffer{items: cp, totalSealed: st.totalSealed}
	}
	packs := make(map[uint64]PackRecord, len(p.packs))
	for k, v := range p.packs {
		packs[k] = v
	}
	packOrder := make([]uint64, len(p.packOrder))
	copy(packOrder, p.packOrder)
	packsByKey := make(map[string]int, len(p.packsByKey))
	for k, v := range p.packsByKey {
		packsByKey[k] = v
	}
	live := make(map[string]ChunkRecord, len(p.live))
	for k, v := range p.live {
		live[k] = v
	}
	return packerSnapshot{
		seq:          p.seq,
		nextPackSeq:  p.nextPackSeq,
		nextChunkSeq: p.nextChunkSeq,
		streams:      streams,
		packs:        packs,
		packOrder:    packOrder,
		packsByKey:   packsByKey,
		live:         live,
		stats:        p.stats,
	}
}

func (p *Packer) restoreLocked(s packerSnapshot) {
	p.seq = s.seq
	p.nextPackSeq = s.nextPackSeq
	p.nextChunkSeq = s.nextChunkSeq
	p.streams = s.streams
	p.packs = s.packs
	p.packOrder = s.packOrder
	p.packsByKey = s.packsByKey
	p.live = s.live
	p.stats = s.stats
	for seq := range p.packs {
		if _, ok := p.pins[seq]; !ok {
			p.pins[seq] = &packPin{}
		}
	}
}

// CheckAccounting verifies every live chunk, buffered chunk, and pack byte
// conservation invariant inside p. Any discrepancy returns a descriptive error.
func (p *Packer) CheckAccounting() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	livePerPack := make(map[uint64]int64, len(p.packs))
	countPerPack := make(map[uint64]int, len(p.packs))
	bufferedPerStream := make([]int64, len(p.streams))

	for k, c := range p.live {
		if c.LogicalKey != k {
			return fmt.Errorf("dogi accounting: live key %q != record key %q", k, c.LogicalKey)
		}
		if c.PackSeq == 0 {
			if c.StreamIdx < 0 || c.StreamIdx >= len(p.streams) {
				return fmt.Errorf("dogi accounting: chunk %q invalid stream %d", k, c.StreamIdx)
			}
			bufferedPerStream[c.StreamIdx] += int64(c.SealedLen)
			continue
		}
		pk, ok := p.packs[c.PackSeq]
		if !ok {
			return fmt.Errorf("dogi accounting: chunk %q references missing pack %d", k, c.PackSeq)
		}
		if pk.Deleted {
			return fmt.Errorf("dogi accounting: live chunk %q points to deleted pack %d", k, c.PackSeq)
		}
		if c.Off+c.SealedLen > uint64(pk.SealedBytes) {
			return fmt.Errorf("dogi accounting: chunk %q range [%d+%d) exceeds pack %d size %d", k, c.Off, c.SealedLen, c.PackSeq, pk.SealedBytes)
		}
		livePerPack[c.PackSeq] += int64(c.SealedLen)
		countPerPack[c.PackSeq]++
	}

	for idx, st := range p.streams {
		var sum int64
		for _, it := range st.items {
			sum += int64(len(it.Frame))
		}
		if sum != st.totalSealed || sum != bufferedPerStream[idx] {
			return fmt.Errorf("dogi accounting: stream %d buffered mismatch sum=%d total=%d live=%d", idx, sum, st.totalSealed, bufferedPerStream[idx])
		}
	}

	var activePackBytes, deletedPackBytes int64
	var deletedCount int64
	activeKeys := make(map[string]int64)
	deletedKeys := make(map[string]int64)
	for _, seq := range p.packOrder {
		pk, ok := p.packs[seq]
		if !ok {
			return fmt.Errorf("dogi accounting: packOrder references missing seq %d", seq)
		}
		if pk.LiveSealedBytes < 0 || pk.DeadSealedBytes < 0 || pk.LiveSealedBytes+pk.DeadSealedBytes != pk.SealedBytes {
			return fmt.Errorf("dogi accounting: pack %d bytes invariant broken live=%d dead=%d total=%d", seq, pk.LiveSealedBytes, pk.DeadSealedBytes, pk.SealedBytes)
		}
		if pk.Deleted {
			if pk.LiveSealedBytes != 0 || pk.LiveChunkCount != 0 || countPerPack[seq] != 0 {
				return fmt.Errorf("dogi accounting: deleted pack %d retains live bytes=%d count=%d", seq, pk.LiveSealedBytes, countPerPack[seq])
			}
			deletedCount++
			deletedKeys[pk.Key] = pk.SealedBytes
			continue
		}
		if livePerPack[seq] != pk.LiveSealedBytes || countPerPack[seq] != pk.LiveChunkCount {
			return fmt.Errorf("dogi accounting: pack %d live mismatch tracked=(%dB,%d) actual=(%dB,%d)",
				seq, pk.LiveSealedBytes, pk.LiveChunkCount, livePerPack[seq], countPerPack[seq])
		}
		activeKeys[pk.Key] = pk.SealedBytes
	}
	for _, sz := range activeKeys {
		activePackBytes += sz
	}
	for k, sz := range deletedKeys {
		if _, stillActive := activeKeys[k]; !stillActive {
			deletedPackBytes += sz
		}
	}

	if int64(len(p.packOrder)) != p.stats.PacksCreated {
		return fmt.Errorf("dogi accounting: pack count %d != PacksCreated %d", len(p.packOrder), p.stats.PacksCreated)
	}
	if deletedCount != p.stats.PacksDeleted {
		return fmt.Errorf("dogi accounting: deleted count %d != PacksDeleted %d", deletedCount, p.stats.PacksDeleted)
	}
	if activePackBytes+p.stats.PhysicalBytesReclaimed != p.stats.PhysicalBytesWritten {
		return fmt.Errorf("dogi accounting: active(%d) + reclaimed(%d) != written(%d)",
			activePackBytes, p.stats.PhysicalBytesReclaimed, p.stats.PhysicalBytesWritten)
	}
	if deletedPackBytes != p.stats.PhysicalBytesReclaimed {
		return fmt.Errorf("dogi accounting: deleted bytes %d != PhysicalBytesReclaimed %d",
			deletedPackBytes, p.stats.PhysicalBytesReclaimed)
	}
	return nil
}

// --- Sealing & binary serialization helpers ---

func prefixedHashKey(kind string, sum [32]byte) string {
	h := hex.EncodeToString(sum[:])
	return kind + "/" + h[:2] + "/" + h
}

func bind(kind string, fields ...[]byte) []byte {
	h := blake3.New()
	_, _ = h.WriteString("hanzoai/vfs dogi seal context")
	frame := func(b []byte) {
		_, _ = h.Write(binary.AppendUvarint(nil, uint64(len(b))))
		_, _ = h.Write(b)
	}
	frame([]byte(kind))
	for _, f := range fields {
		frame(f)
	}
	return h.Sum(nil)
}

func headAAD(place string) []byte { return bind("dogi-head", []byte(place)) }

func manifestAAD(volID [16]byte, seq uint64) []byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], seq)
	return bind("dogi-manifest", volID[:], b[:])
}

func chunkAAD(volID [16]byte, plainHash [32]byte) []byte {
	return bind("dogi-chunk", volID[:], plainHash[:])
}

func checkCipher(c Cipher) error {
	probe := make([]byte, 64)
	if _, err := rand.Read(probe); err != nil {
		return err
	}
	sealed, id, err := c.Seal(append([]byte(nil), probe...))
	if err != nil {
		return fmt.Errorf("%w: seal: %v", ErrBadCipher, err)
	}
	if bytes.Contains(sealed, probe) || bytes.Contains(sealed, probe[:16]) {
		return fmt.Errorf("%w: output contains plaintext", ErrBadCipher)
	}
	got, err := c.Open(append([]byte(nil), sealed...), id)
	if err != nil || !bytes.Equal(got, probe) {
		return fmt.Errorf("%w: open mismatch: %v", ErrBadCipher, err)
	}
	altered := append([]byte(nil), sealed...)
	altered[len(altered)/2] ^= 1
	if _, err := c.Open(altered, id); err == nil {
		return fmt.Errorf("%w: opened altered ciphertext", ErrBadCipher)
	}
	return nil
}

func sealFrame(c Cipher, aad, plain []byte) ([]byte, error) {
	if len(aad) != bindLen {
		return nil, fmt.Errorf("dogi: seal context %d bytes, want %d", len(aad), bindLen)
	}
	buf := make([]byte, 0, len(aad)+len(plain))
	buf = append(buf, aad...)
	buf = append(buf, plain...)
	sealed, id, err := c.Seal(buf)
	if err != nil {
		return nil, fmt.Errorf("dogi: seal: %w", err)
	}
	if bytes.Contains(sealed, aad) {
		return nil, fmt.Errorf("%w: output contains aad", ErrBadCipher)
	}
	if len(id) == 0 || len(id) > 255 {
		return nil, fmt.Errorf("dogi: invalid key id length %d", len(id))
	}
	out := make([]byte, 0, 1+len(id)+len(sealed))
	out = append(out, byte(len(id)))
	out = append(out, id...)
	return append(out, sealed...), nil
}

func unsealFrame(c Cipher, aad, frame []byte) ([]byte, error) {
	if len(aad) != bindLen {
		return nil, fmt.Errorf("dogi: unseal context %d bytes, want %d", len(aad), bindLen)
	}
	if len(frame) < 1 || len(frame) < 1+int(frame[0]) {
		return nil, fmt.Errorf("%w: truncated frame", ErrCorrupt)
	}
	n := 1 + int(frame[0])
	plain, err := c.Open(frame[n:], string(frame[1:n]))
	if err != nil {
		return nil, err
	}
	if len(plain) < bindLen || subtle.ConstantTimeCompare(plain[:bindLen], aad) != 1 {
		return nil, fmt.Errorf("%w: sealing context mismatch", ErrCorrupt)
	}
	return plain[bindLen:], nil
}

type checkpointHead struct {
	VolumeID    [16]byte
	Seq         uint64
	Mode        Mode
	PackTarget  uint64
	ManifestRef [32]byte
	Stats       Stats
}

func (h checkpointHead) marshal() []byte {
	b := make([]byte, 0, 192)
	b = append(b, headMagic...)
	b = append(b, h.VolumeID[:]...)
	b = binary.AppendUvarint(b, h.Seq)
	b = append(b, byte(h.Mode))
	b = binary.AppendUvarint(b, h.PackTarget)
	b = append(b, h.ManifestRef[:]...)
	b = appendStats(b, h.Stats)
	sum := blake3.Sum(b)
	return append(b, sum[:]...)
}

func decodeHead(c Cipher, place string, sealed []byte) (checkpointHead, error) {
	plain, err := unsealFrame(c, headAAD(place), sealed)
	if err != nil {
		return checkpointHead{}, err
	}
	if len(plain) < len(headMagic)+16+32+32 || string(plain[:len(headMagic)]) != headMagic {
		return checkpointHead{}, fmt.Errorf("%w: invalid head header", ErrCorrupt)
	}
	body, wantSum := plain[:len(plain)-32], plain[len(plain)-32:]
	gotSum := blake3.Sum(body)
	if !bytes.Equal(gotSum[:], wantSum) {
		return checkpointHead{}, fmt.Errorf("%w: head checksum mismatch", ErrCorrupt)
	}
	cur := body[len(headMagic):]
	var h checkpointHead
	if len(cur) < 16 {
		return checkpointHead{}, ErrCorrupt
	}
	copy(h.VolumeID[:], cur[:16])
	cur = cur[16:]
	var ok bool
	if h.Seq, cur, ok = takeUvarint(cur); !ok || len(cur) < 1 {
		return checkpointHead{}, ErrCorrupt
	}
	h.Mode = Mode(cur[0])
	cur = cur[1:]
	if h.PackTarget, cur, ok = takeUvarint(cur); !ok || len(cur) < 32 {
		return checkpointHead{}, ErrCorrupt
	}
	copy(h.ManifestRef[:], cur[:32])
	cur = cur[32:]
	h.Stats, cur, ok = takeStats(cur)
	if !ok || len(cur) != 0 {
		return checkpointHead{}, ErrCorrupt
	}
	return h, nil
}

type checkpointManifest struct {
	VolumeID     [16]byte
	Seq          uint64
	NextPackSeq  uint64
	NextChunkSeq uint64
	Packs        []PackRecord
	Live         []ChunkRecord
	Stats        Stats
}

func (p *Packer) buildManifestLocked(seq uint64) checkpointManifest {
	packs := make([]PackRecord, 0, len(p.packOrder))
	for _, pSeq := range p.packOrder {
		if pk, ok := p.packs[pSeq]; ok {
			packs = append(packs, pk)
		}
	}
	live := make([]ChunkRecord, 0, len(p.live))
	for _, c := range p.live {
		live = append(live, c)
	}
	sort.Slice(live, func(i, j int) bool { return live[i].LogicalKey < live[j].LogicalKey })
	return checkpointManifest{
		VolumeID:     p.volID,
		Seq:          seq,
		NextPackSeq:  p.nextPackSeq,
		NextChunkSeq: p.nextChunkSeq,
		Packs:        packs,
		Live:         live,
		Stats:        p.stats,
	}
}

func (m checkpointManifest) marshal() []byte {
	b := make([]byte, 0, 256+len(m.Packs)*128+len(m.Live)*96)
	b = append(b, manifestMagic...)
	b = append(b, m.VolumeID[:]...)
	b = binary.AppendUvarint(b, m.Seq)
	b = binary.AppendUvarint(b, m.NextPackSeq)
	b = binary.AppendUvarint(b, m.NextChunkSeq)
	b = binary.AppendUvarint(b, uint64(len(m.Packs)))
	for _, pk := range m.Packs {
		b = binary.AppendUvarint(b, pk.Seq)
		b = binary.AppendUvarint(b, uint64(len(pk.Key)))
		b = append(b, pk.Key...)
		b = append(b, pk.BLAKE3[:]...)
		b = binary.AppendUvarint(b, uint64(pk.StreamIdx))
		b = append(b, byte(pk.PredictedClass))
		b = binary.AppendUvarint(b, uint64(pk.SealedBytes))
		b = binary.AppendUvarint(b, uint64(pk.LiveSealedBytes))
		b = binary.AppendUvarint(b, uint64(pk.DeadSealedBytes))
		b = binary.AppendUvarint(b, uint64(pk.ChunkCount))
		b = binary.AppendUvarint(b, uint64(pk.LiveChunkCount))
		if pk.Deleted {
			b = append(b, 1)
		} else {
			b = append(b, 0)
		}
	}
	b = binary.AppendUvarint(b, uint64(len(m.Live)))
	for _, c := range m.Live {
		b = binary.AppendUvarint(b, uint64(len(c.LogicalKey)))
		b = append(b, c.LogicalKey...)
		b = append(b, c.PlainHash[:]...)
		b = binary.AppendUvarint(b, c.PlainLen)
		b = binary.AppendUvarint(b, c.PackSeq)
		b = binary.AppendUvarint(b, c.Off)
		b = binary.AppendUvarint(b, c.SealedLen)
		b = append(b, byte(c.Class))
		b = binary.AppendUvarint(b, c.Version)
	}
	b = appendStats(b, m.Stats)
	sum := blake3.Sum(b)
	return append(b, sum[:]...)
}

func fetchManifest(ctx context.Context, be backend.Backend, c Cipher, hd checkpointHead) (checkpointManifest, error) {
	manKey := prefixedHashKey("meta", hd.ManifestRef)
	sealed, _, err := be.Get(ctx, manKey, maxManifestReadBytes)
	if err != nil {
		return checkpointManifest{}, fmt.Errorf("dogi: fetch manifest %s: %w", manKey, err)
	}
	if blake3.Sum(sealed) != hd.ManifestRef {
		return checkpointManifest{}, fmt.Errorf("%w: manifest %s hash mismatch", ErrCorrupt, manKey)
	}
	plain, err := unsealFrame(c, manifestAAD(hd.VolumeID, hd.Seq), sealed)
	if err != nil {
		return checkpointManifest{}, err
	}
	return decodeManifest(plain, hd)
}

func decodeManifest(plain []byte, hd checkpointHead) (checkpointManifest, error) {
	if len(plain) < len(manifestMagic)+16+32 || string(plain[:len(manifestMagic)]) != manifestMagic {
		return checkpointManifest{}, fmt.Errorf("%w: invalid manifest header", ErrCorrupt)
	}
	body, wantSum := plain[:len(plain)-32], plain[len(plain)-32:]
	gotSum := blake3.Sum(body)
	if !bytes.Equal(gotSum[:], wantSum) {
		return checkpointManifest{}, fmt.Errorf("%w: manifest checksum mismatch", ErrCorrupt)
	}
	cur := body[len(manifestMagic):]
	var m checkpointManifest
	copy(m.VolumeID[:], cur[:16])
	cur = cur[16:]
	if m.VolumeID != hd.VolumeID {
		return checkpointManifest{}, fmt.Errorf("%w: manifest volume id mismatch", ErrCorrupt)
	}
	var ok bool
	if m.Seq, cur, ok = takeUvarint(cur); !ok || m.Seq != hd.Seq {
		return checkpointManifest{}, ErrCorrupt
	}
	if m.NextPackSeq, cur, ok = takeUvarint(cur); !ok {
		return checkpointManifest{}, ErrCorrupt
	}
	if m.NextChunkSeq, cur, ok = takeUvarint(cur); !ok {
		return checkpointManifest{}, ErrCorrupt
	}
	nPacks, cur, ok := takeUvarint(cur)
	if !ok || nPacks > 1<<24 {
		return checkpointManifest{}, ErrCorrupt
	}
	m.Packs = make([]PackRecord, nPacks)
	for i := range m.Packs {
		var pk PackRecord
		var keyLen, streamIdx, sb, lb, db, cc, lcc uint64
		if pk.Seq, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if keyLen, cur, ok = takeUvarint(cur); !ok || keyLen > uint64(len(cur)) {
			return checkpointManifest{}, ErrCorrupt
		}
		pk.Key = string(cur[:keyLen])
		cur = cur[keyLen:]
		if len(cur) < 32 {
			return checkpointManifest{}, ErrCorrupt
		}
		copy(pk.BLAKE3[:], cur[:32])
		cur = cur[32:]
		if streamIdx, cur, ok = takeUvarint(cur); !ok || len(cur) < 1 {
			return checkpointManifest{}, ErrCorrupt
		}
		pk.StreamIdx = int(streamIdx)
		pk.PredictedClass = Lifetime(cur[0])
		cur = cur[1:]
		if sb, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if lb, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if db, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if cc, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if lcc, cur, ok = takeUvarint(cur); !ok || len(cur) < 1 {
			return checkpointManifest{}, ErrCorrupt
		}
		pk.SealedBytes = int64(sb)
		pk.LiveSealedBytes = int64(lb)
		pk.DeadSealedBytes = int64(db)
		pk.ChunkCount = int(cc)
		pk.LiveChunkCount = int(lcc)
		pk.Deleted = cur[0] == 1
		cur = cur[1:]
		m.Packs[i] = pk
	}

	nLive, cur, ok := takeUvarint(cur)
	if !ok || nLive > 1<<26 {
		return checkpointManifest{}, ErrCorrupt
	}
	m.Live = make([]ChunkRecord, nLive)
	for i := range m.Live {
		var c ChunkRecord
		var kl uint64
		if kl, cur, ok = takeUvarint(cur); !ok || kl > uint64(len(cur)) {
			return checkpointManifest{}, ErrCorrupt
		}
		c.LogicalKey = string(cur[:kl])
		cur = cur[kl:]
		if len(cur) < 32 {
			return checkpointManifest{}, ErrCorrupt
		}
		copy(c.PlainHash[:], cur[:32])
		cur = cur[32:]
		if c.PlainLen, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if c.PackSeq, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if c.Off, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		if c.SealedLen, cur, ok = takeUvarint(cur); !ok || len(cur) < 1 {
			return checkpointManifest{}, ErrCorrupt
		}
		c.Class = Lifetime(cur[0])
		cur = cur[1:]
		if c.Version, cur, ok = takeUvarint(cur); !ok {
			return checkpointManifest{}, ErrCorrupt
		}
		m.Live[i] = c
	}

	m.Stats, cur, ok = takeStats(cur)
	if !ok || len(cur) != 0 {
		return checkpointManifest{}, ErrCorrupt
	}
	return m, nil
}

func appendStats(b []byte, s Stats) []byte {
	vals := []int64{
		s.LogicalChunksWritten,
		s.LogicalPlaintextBytes,
		s.LogicalSealedBytes,
		s.PhysicalBytesWritten,
		s.PhysicalBytesReclaimed,
		s.LiveBytesCopied,
		s.PacksCreated,
		s.PacksDeleted,
		s.LostACKRecoveries,
		s.PackDedupReuses,
		s.Rollbacks,
	}
	for _, v := range vals {
		b = binary.AppendVarint(b, v)
	}
	return b
}

func takeStats(cur []byte) (Stats, []byte, bool) {
	var vals [11]int64
	for i := range vals {
		v, n := binary.Varint(cur)
		if n <= 0 {
			return Stats{}, nil, false
		}
		vals[i] = v
		cur = cur[n:]
	}
	return Stats{
		LogicalChunksWritten:   vals[0],
		LogicalPlaintextBytes:  vals[1],
		LogicalSealedBytes:     vals[2],
		PhysicalBytesWritten:   vals[3],
		PhysicalBytesReclaimed: vals[4],
		LiveBytesCopied:        vals[5],
		PacksCreated:           vals[6],
		PacksDeleted:           vals[7],
		LostACKRecoveries:      vals[8],
		PackDedupReuses:        vals[9],
		Rollbacks:              vals[10],
	}, cur, true
}

func takeUvarint(cur []byte) (uint64, []byte, bool) {
	v, n := binary.Uvarint(cur)
	if n <= 0 {
		return 0, nil, false
	}
	return v, cur[n:], true
}

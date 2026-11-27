// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package dogi

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/hanzoai/vfs/internal/blake3"
	"github.com/hanzoai/vfs/pkg/fence"
)

// CompactionPolicy controls deterministic pack reclamation during Compact.
type CompactionPolicy struct {
	// GarbageRatio is the minimum dead-byte fraction [0.0, 1.0] required for
	// an existing pack to be reclaimed. If zero, any pack with DeadSealedBytes > 0
	// is eligible.
	GarbageRatio float64

	// MinDeadBytes optionally skips packs with fewer than MinDeadBytes dead bytes,
	// unless the pack is 100% dead.
	MinDeadBytes int64

	// MaxPacks caps the number of candidate packs reclaimed in a single Compact
	// call (0 means unlimited).
	MaxPacks int
}

// CompactionResult summarizes the deterministic work done by one Compact call.
type CompactionResult struct {
	PacksInspected         int           `json:"packs_inspected"`
	PacksCompacted         int           `json:"packs_compacted"`
	PacksDeleted           int           `json:"packs_deleted"`
	ChunksRelocated        int           `json:"chunks_relocated"`
	LiveBytesCopied        int64         `json:"live_bytes_copied"`
	PhysicalBytesWritten   int64         `json:"physical_bytes_written"`
	PhysicalBytesReclaimed int64         `json:"physical_bytes_reclaimed"`
	Duration               time.Duration `json:"duration"`
}

type relocationSource struct {
	packSeq uint64
	packKey string
	chunks  []ChunkRecord
}

type relocationTarget struct {
	srcPackSeq uint64
	logicalKey string
	version    uint64
	streamIdx  int
	class      Lifetime
	plainHash  [32]byte
	plainLen   uint64
	frame      []byte
}

// Compact performs deterministic pack reclamation in ascending packSeq order:
//
//  1. Selects packs whose DeadSealedBytes / SealedBytes >= policy.GarbageRatio.
//  2. Reads and verifies surviving live chunks outside p.mu while pinning
//     source packs against deletion.
//  3. Packs surviving frames into new pack/<2hex>/<blake3> objects and uploads
//     them FIRST via immutable PutIfAbsent.
//  4. Atomically updates live chunk locations and commits an updated "head"
//     checkpoint manifest via pkg/fence.Store BEFORE deleting old dead packs.
//  5. Waits for all pinned concurrent Get readers on each retired pack to
//     drain, then deletes the dead pack from the backend.
func (p *Packer) Compact(ctx context.Context, policy CompactionPolicy) (CompactionResult, error) {
	if ctx == nil {
		return CompactionResult{}, errors.New("dogi: nil context")
	}
	if p.readOnly {
		return CompactionResult{}, ErrReadOnly
	}
	if policy.GarbageRatio < 0 || policy.GarbageRatio > 1 {
		return CompactionResult{}, fmt.Errorf("dogi: invalid GarbageRatio %v, want [0, 1]", policy.GarbageRatio)
	}
	start := time.Now()

	p.wmu.Lock()
	defer p.wmu.Unlock()

	p.mu.Lock()
	if p.fenced {
		p.mu.Unlock()
		return CompactionResult{}, fence.ErrStaleRound
	}
	snap := p.snapshotLocked()

	candidates, inspected := p.selectCompactionCandidatesLocked(policy)
	if len(candidates) == 0 {
		p.mu.Unlock()
		return CompactionResult{PacksInspected: inspected, Duration: time.Since(start)}, nil
	}

	// Pin all candidate source packs while reading surviving chunks outside p.mu.
	for _, cand := range candidates {
		if pin := p.pins[cand.packSeq]; pin != nil {
			pin.readers++
		}
	}
	p.mu.Unlock()

	unpinCandidates := func() {
		p.mu.Lock()
		for _, cand := range candidates {
			if pin := p.pins[cand.packSeq]; pin != nil {
				pin.readers--
				if pin.readers == 0 && pin.draining {
					p.pinCond.Broadcast()
				}
			}
		}
		p.mu.Unlock()
	}

	relocated, copiedBytes, err := p.readAndVerifySurvivors(ctx, candidates)
	unpinCandidates()
	if err != nil {
		return CompactionResult{}, err
	}

	// Pack survivors into dedicated relocation streams and upload BEFORE updating head or deleting old packs.
	uploadedPacks, err := p.buildAndUploadRelocatedPacks(ctx, relocated)
	if err != nil {
		p.mu.Lock()
		p.restoreLocked(snap)
		p.stats.Rollbacks++
		p.mu.Unlock()
		return CompactionResult{}, err
	}

	var res CompactionResult
	res.PacksInspected = inspected
	res.PacksCompacted = len(candidates)
	res.ChunksRelocated = len(relocated)
	res.LiveBytesCopied = copiedBytes

	p.mu.Lock()
	for _, up := range uploadedPacks {
		p.registerRelocationPackLocked(up, &res)
	}
	// Mark drained candidates ready for manifest update; keep Deleted=false until backend Delete succeeds.
	for _, cand := range candidates {
		pk := p.packs[cand.packSeq]
		pk.LiveSealedBytes = 0
		pk.DeadSealedBytes = pk.SealedBytes
		pk.LiveChunkCount = 0
		p.packs[cand.packSeq] = pk
		if pin := p.pins[cand.packSeq]; pin != nil {
			pin.draining = true
		}
	}
	p.stats.LiveBytesCopied += copiedBytes
	nextSeq := p.seq + 1
	man := p.buildManifestLocked(nextSeq)
	// Also record candidates as deleted in the new committed manifest BEFORE deleting
	// them on the backend so a crash after head commit never leaves references to dead packs.
	candSet := make(map[uint64]struct{}, len(candidates))
	for _, cand := range candidates {
		candSet[cand.packSeq] = struct{}{}
	}
	// Track unique physical keys that become completely unreferenced so deduped keys
	// are only credited to PhysicalBytesReclaimed once.
	simKeyRefs := make(map[string]int, len(p.packsByKey))
	for k, v := range p.packsByKey {
		simKeyRefs[k] = v
	}
	var reclaimedBytes int64
	for i := range man.Packs {
		if _, isDead := candSet[man.Packs[i].Seq]; isDead {
			man.Packs[i].Deleted = true
			man.Stats.PacksDeleted++
			simKeyRefs[man.Packs[i].Key]--
			if simKeyRefs[man.Packs[i].Key] <= 0 {
				reclaimedBytes += man.Packs[i].SealedBytes
			}
		}
	}
	man.Stats.PhysicalBytesReclaimed += reclaimedBytes
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
		return CompactionResult{}, err
	}

	// Head has durably advanced to nextSeq; advance in-memory seq and commit status immediately.
	p.mu.Lock()
	p.seq = nextSeq
	for _, seq := range p.packOrder {
		if pk, ok := p.packs[seq]; ok && !pk.Committed {
			pk.Committed = true
			p.packs[seq] = pk
		}
	}
	p.mu.Unlock()

	// Wait for any in-flight concurrent Get readers pinned on retiring packs, then delete from backend.
	stopAfter := context.AfterFunc(ctx, func() {
		p.mu.Lock()
		p.pinCond.Broadcast()
		p.mu.Unlock()
	})
	defer stopAfter()

	for _, cand := range candidates {
		if err := p.waitForPinDrain(ctx, cand.packSeq); err != nil {
			// Head already points to new packs; old pack remains harmlessly unreferenced until retry.
			return res, err
		}
		p.mu.Lock()
		p.packsByKey[cand.packKey]--
		shouldDelete := p.packsByKey[cand.packKey] <= 0
		if shouldDelete {
			delete(p.packsByKey, cand.packKey)
		}
		p.mu.Unlock()

		if shouldDelete {
			if err := p.be.Delete(ctx, cand.packKey); err != nil {
				return res, fmt.Errorf("dogi: delete compacted pack %s: %w", cand.packKey, err)
			}
		}

		p.mu.Lock()
		pk := p.packs[cand.packSeq]
		pk.Deleted = true
		pk.Committed = true
		p.packs[cand.packSeq] = pk
		delete(p.pins, cand.packSeq)
		p.stats.PacksDeleted++
		if shouldDelete {
			p.stats.PhysicalBytesReclaimed += pk.SealedBytes
			res.PhysicalBytesReclaimed += pk.SealedBytes
		}
		res.PacksDeleted++
		p.mu.Unlock()
	}

	res.Duration = time.Since(start)
	return res, nil
}

func (p *Packer) selectCompactionCandidatesLocked(policy CompactionPolicy) ([]relocationSource, int) {
	chunksByPack := make(map[uint64][]ChunkRecord)
	for _, c := range p.live {
		if c.PackSeq > 0 {
			chunksByPack[c.PackSeq] = append(chunksByPack[c.PackSeq], c)
		}
	}

	// Deterministic ascending packSeq order.
	seqs := make([]uint64, 0, len(p.packs))
	for seq, pk := range p.packs {
		if !pk.Deleted && pk.SealedBytes > 0 {
			seqs = append(seqs, seq)
		}
	}
	sort.Slice(seqs, func(i, j int) bool { return seqs[i] < seqs[j] })

	var candidates []relocationSource
	for _, seq := range seqs {
		pk := p.packs[seq]
		if pk.DeadSealedBytes == 0 {
			continue
		}
		fullyDead := pk.DeadSealedBytes == pk.SealedBytes
		if !fullyDead && pk.DeadSealedBytes < policy.MinDeadBytes {
			continue
		}
		ratio := float64(pk.DeadSealedBytes) / float64(pk.SealedBytes)
		if policy.GarbageRatio > 0 && ratio+1e-12 < policy.GarbageRatio {
			continue
		}
		ch := chunksByPack[seq]
		sort.Slice(ch, func(i, j int) bool { return ch[i].Off < ch[j].Off })
		candidates = append(candidates, relocationSource{
			packSeq: seq,
			packKey: pk.Key,
			chunks:  ch,
		})
		if policy.MaxPacks > 0 && len(candidates) >= policy.MaxPacks {
			break
		}
	}
	return candidates, len(seqs)
}

func (p *Packer) readAndVerifySurvivors(ctx context.Context, candidates []relocationSource) ([]relocationTarget, int64, error) {
	var targets []relocationTarget
	var copiedBytes int64
	for _, cand := range candidates {
		for _, c := range cand.chunks {
			frame, err := p.be.GetRange(ctx, cand.packKey, int64(c.Off), int64(c.SealedLen))
			if err != nil {
				return nil, 0, fmt.Errorf("dogi: compact read %s from %s: %w", c.LogicalKey, cand.packKey, err)
			}
			plain, err := unsealFrame(p.cipher, chunkAAD(p.volID, c.PlainHash), frame)
			if err != nil {
				return nil, 0, fmt.Errorf("dogi: compact verify %s: %w", c.LogicalKey, err)
			}
			if uint64(len(plain)) != c.PlainLen || blake3.Sum(plain) != c.PlainHash {
				return nil, 0, fmt.Errorf("%w: compact hash mismatch for %s", ErrCorrupt, c.LogicalKey)
			}

			targetClass := c.Class
			targetStream := 0
			if p.cfg.Mode == ModeLifetimeSeparated {
				if !p.cfg.DisableSurvivorPromotion && targetClass == LifetimeEphemeral {
					// Surviving chunks from reclaimed ephemeral packs are promoted so
					// mispredicted long-lived objects are not repeatedly churned.
					targetClass = LifetimeLong
				} else if !p.cfg.DisableSurvivorPromotion && targetClass == LifetimeMedium {
					targetClass = LifetimeLong
				}
				targetStream = int(targetClass)
			}
			targets = append(targets, relocationTarget{
				srcPackSeq: cand.packSeq,
				logicalKey: c.LogicalKey,
				version:    c.Version,
				streamIdx:  targetStream,
				class:      targetClass,
				plainHash:  c.PlainHash,
				plainLen:   c.PlainLen,
				frame:      frame,
			})
			copiedBytes += int64(len(frame))
		}
	}
	return targets, copiedBytes, nil
}

type relocationPackUpload struct {
	streamIdx int
	class     Lifetime
	key       string
	hash      [32]byte
	payload   []byte
	items     []relocationTarget
	deduped   bool
	recovered bool
}

func (p *Packer) buildAndUploadRelocatedPacks(ctx context.Context, targets []relocationTarget) ([]relocationPackUpload, error) {
	if len(targets) == 0 {
		return nil, nil
	}
	nStreams := p.cfg.Mode.NumStreams()
	buckets := make([][]relocationTarget, nStreams)
	sizes := make([]int64, nStreams)
	var uploads []relocationPackUpload

	flushBucket := func(streamIdx int) error {
		items := buckets[streamIdx]
		if len(items) == 0 {
			return nil
		}
		payload := make([]byte, 0, sizes[streamIdx])
		for _, it := range items {
			payload = append(payload, it.frame...)
		}
		hash := blake3.Sum(payload)
		key := prefixedHashKey("pack", hash)
		deduped, recovered, err := p.putImmutable(ctx, key, payload)
		if err != nil {
			return err
		}
		cls := LifetimeMedium
		if p.cfg.Mode == ModeLifetimeSeparated && streamIdx >= 0 && streamIdx < NumLifetimes {
			cls = Lifetime(streamIdx)
		}
		uploads = append(uploads, relocationPackUpload{
			streamIdx: streamIdx,
			class:     cls,
			key:       key,
			hash:      hash,
			payload:   payload,
			items:     items,
			deduped:   deduped,
			recovered: recovered,
		})
		buckets[streamIdx] = nil
		sizes[streamIdx] = 0
		return nil
	}

	for _, t := range targets {
		sIdx := t.streamIdx
		if sIdx < 0 || sIdx >= nStreams {
			sIdx = 0
		}
		buckets[sIdx] = append(buckets[sIdx], t)
		sizes[sIdx] += int64(len(t.frame))
		if sizes[sIdx] >= int64(p.cfg.PackTarget) {
			if err := flushBucket(sIdx); err != nil {
				return nil, err
			}
		}
	}
	for sIdx := range buckets {
		if err := flushBucket(sIdx); err != nil {
			return nil, err
		}
	}
	return uploads, nil
}

func (p *Packer) registerRelocationPackLocked(up relocationPackUpload, res *CompactionResult) {
	p.nextPackSeq++
	newSeq := p.nextPackSeq
	alreadyInStore := p.packsByKey[up.key] > 0
	p.packsByKey[up.key]++
	p.pins[newSeq] = &packPin{}

	var off uint64
	var liveBytes int64
	var liveCount int
	for _, it := range up.items {
		n := uint64(len(it.frame))
		if cur, ok := p.live[it.logicalKey]; ok && cur.Version == it.version && cur.PackSeq == it.srcPackSeq {
			cur.PackSeq = newSeq
			cur.StreamIdx = up.streamIdx
			cur.Off = off
			cur.Class = it.class
			p.live[it.logicalKey] = cur
			liveBytes += int64(n)
			liveCount++
		}
		off += n
	}

	sealedTotal := int64(len(up.payload))
	p.packs[newSeq] = PackRecord{
		Seq:             newSeq,
		Key:             up.key,
		BLAKE3:          up.hash,
		StreamIdx:       up.streamIdx,
		PredictedClass:  up.class,
		SealedBytes:     sealedTotal,
		LiveSealedBytes: liveBytes,
		DeadSealedBytes: sealedTotal - liveBytes,
		ChunkCount:      len(up.items),
		LiveChunkCount:  liveCount,
		Committed:       false,
	}
	p.packOrder = append(p.packOrder, newSeq)
	p.stats.PacksCreated++
	if up.recovered {
		p.stats.LostACKRecoveries++
	}
	if up.deduped && alreadyInStore {
		p.stats.PackDedupReuses++
	} else {
		p.stats.PhysicalBytesWritten += sealedTotal
		res.PhysicalBytesWritten += sealedTotal
	}
}

func (p *Packer) waitForPinDrain(ctx context.Context, packSeq uint64) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	for {
		pin := p.pins[packSeq]
		if pin == nil || pin.readers == 0 {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		p.pinCond.Wait()
	}
}

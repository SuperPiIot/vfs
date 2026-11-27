// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package vfs

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hanzoai/vfs/internal/blake3"
	"github.com/hanzoai/vfs/pkg/backend"
	"github.com/hanzoai/vfs/pkg/fence"
)

const (
	defaultParaSyncStreams = 8
	// maxParaSyncStagedShardBytes bounds how many unpruned source shard ciphertexts
	// RestoreDelta keeps in memory between planning and destination upload so small
	// and medium restores avoid double-fetching shards without unbounded RAM growth.
	maxParaSyncStagedShardBytes = 128 << 20
)

// DeltaPlan describes the verified minimal object-level transfer required to
// restore source's committed checkpoint onto destination without modifying the
// canonical format-v2 layout or re-encrypting pack payloads.
type DeltaPlan struct {
	SrcHead           Head
	DstHead           Head
	HasDst            bool
	ShardsToCopy      []string
	ShardsPruned      int
	PacksToCopy       []string
	PacksPruned       int
	TotalSourceShards int
	TotalSourcePacks  int
	MetaBytes         int64
	PackBytes         int64
	FullMetaBytes     int64
	FullPackBytes     int64
	FullSnapshotBytes int64
}

// RestoreOptions configures a ParaSync restore execution.
type RestoreOptions struct {
	// Round claims the destination Head at this round (must exceed the
	// destination's recorded round, and be in (0, 2^64-1)).
	Round uint64
	// Streams bounds concurrent backend GET/Stat/PutIfAbsent operations (default 8).
	Streams int
}

// RestoreStats reports measured counts, transferred bytes, resource usage, and
// timings from a RestoreDelta execution.
type RestoreStats struct {
	Plan              DeltaPlan
	ShardsCopied      int
	PacksCopied       int
	BytesTransferred  int64
	FullSnapshotBytes int64
	SourceGets        int64
	DestPuts          int64
	DestStats         int64
	Duration          time.Duration
	CPUTime           time.Duration
	PeakGoroutines    int
}

// parasyncCountingBackend wraps a backend.Backend to measure real storage calls
// and transferred payload bytes while preserving String() for headAAD sealing.
type parasyncCountingBackend struct {
	backend.Backend
	gets          atomic.Int64
	getRanges     atomic.Int64
	stats         atomic.Int64
	puts          atomic.Int64
	putIfAbsents  atomic.Int64
	putIfVersions atomic.Int64
	deletes       atomic.Int64
	bytesRead     atomic.Int64
	bytesWritten  atomic.Int64
}

func newParaSyncCountingBackend(be backend.Backend) *parasyncCountingBackend {
	return &parasyncCountingBackend{Backend: be}
}

func (b *parasyncCountingBackend) Get(ctx context.Context, key string, max int64) ([]byte, backend.Version, error) {
	b.gets.Add(1)
	data, ver, err := b.Backend.Get(ctx, key, max)
	if err == nil {
		b.bytesRead.Add(int64(len(data)))
	}
	return data, ver, err
}

func (b *parasyncCountingBackend) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	b.getRanges.Add(1)
	data, err := b.Backend.GetRange(ctx, key, off, n)
	if err == nil {
		b.bytesRead.Add(int64(len(data)))
	}
	return data, err
}

func (b *parasyncCountingBackend) Put(ctx context.Context, key string, data []byte) error {
	b.puts.Add(1)
	err := b.Backend.Put(ctx, key, data)
	if err == nil {
		b.bytesWritten.Add(int64(len(data)))
	}
	return err
}

func (b *parasyncCountingBackend) PutIfAbsent(ctx context.Context, key string, data []byte) (backend.Version, error) {
	b.puts.Add(1)
	b.putIfAbsents.Add(1)
	ver, err := b.Backend.PutIfAbsent(ctx, key, data)
	if err == nil {
		b.bytesWritten.Add(int64(len(data)))
	}
	return ver, err
}

func (b *parasyncCountingBackend) PutIfVersion(ctx context.Context, key string, data []byte, v backend.Version) (backend.Version, error) {
	b.puts.Add(1)
	b.putIfVersions.Add(1)
	ver, err := b.Backend.PutIfVersion(ctx, key, data, v)
	if err == nil {
		b.bytesWritten.Add(int64(len(data)))
	}
	return ver, err
}

func (b *parasyncCountingBackend) Delete(ctx context.Context, key string) error {
	b.deletes.Add(1)
	return b.Backend.Delete(ctx, key)
}

func (b *parasyncCountingBackend) Stat(ctx context.Context, key string) (int64, error) {
	b.stats.Add(1)
	return b.Backend.Stat(ctx, key)
}

type dstNodeInfo struct {
	level            int
	sealedLen        int64
	children         []ref
	packs            map[packID]uint64
	subtreeShards    int
	subtreeMetaBytes int64
}

type planExtras struct {
	dstRecordedRound uint64
	shardsToCopyRefs map[string]ref
	stagedShards     map[string][]byte
	packSizes        map[string]int64
}

// PlanDelta inspects source and destination checkpoints, verifies every visited
// shard against its BLAKE3 content address and volume sealing context, prunes
// subtrees already reachable in destination in O(1), and determines the exact
// shards and packs required to restore source onto destination.
func PlanDelta(ctx context.Context, src, dst backend.Backend, c Cipher) (*DeltaPlan, error) {
	if ctx == nil {
		return nil, errors.New("vfs: parasync requires a non-nil context")
	}
	if src == nil || dst == nil || c == nil {
		return nil, errors.New("vfs: parasync requires source backend, destination backend, and cipher")
	}
	if err := check(c); err != nil {
		return nil, err
	}
	var peakG atomic.Int64
	plan, _, err := planDelta(ctx, src, dst, c, defaultParaSyncStreams, false, &peakG)
	return plan, err
}

// RestoreDelta restores source's committed checkpoint onto destination using a
// parallel delta transfer (or full parallel restore when destination has no
// head), verifies the full restored tree on destination before touching head,
// and atomically advances destination head with round fencing.
func RestoreDelta(ctx context.Context, src, dst backend.Backend, c Cipher, opts RestoreOptions) (*RestoreStats, error) {
	if ctx == nil {
		return nil, errors.New("vfs: parasync requires a non-nil context")
	}
	if src == nil || dst == nil || c == nil {
		return nil, errors.New("vfs: parasync requires source backend, destination backend, and cipher")
	}
	if opts.Round == 0 || opts.Round == math.MaxUint64 {
		return nil, fmt.Errorf("%w: RestoreDelta needs a Round above 0 and below 2^64-1", fs.ErrInvalid)
	}
	if err := check(c); err != nil {
		return nil, err
	}

	streams := opts.Streams
	if streams <= 0 {
		streams = defaultParaSyncStreams
	}

	startWall := time.Now()
	startCPU := parasyncProcessCPU()
	var peakG atomic.Int64
	recordPeakGoroutines(&peakG)

	srcCount := newParaSyncCountingBackend(src)
	dstCount := newParaSyncCountingBackend(dst)

	plan, ext, err := planDelta(ctx, srcCount, dstCount, c, streams, true, &peakG)
	if err != nil {
		return nil, err
	}
	if plan.HasDst && opts.Round <= ext.dstRecordedRound {
		return nil, fmt.Errorf("%w: round %d does not exceed destination recorded round %d", ErrStaleRound, opts.Round, ext.dstRecordedRound)
	}

	var (
		packsCopied      atomic.Int64
		shardsCopied     atomic.Int64
		bytesTransferred atomic.Int64
	)

	// Canonical VFS write order (LLM.md Rule 3): packs first, then shards, then head.
	if err := runParallel(ctx, streams, len(plan.PacksToCopy), &peakG, func(wctx context.Context, idx int) error {
		pk := plan.PacksToCopy[idx]
		expectedSize := ext.packSizes[pk]
		data, _, err := srcCount.Get(wctx, pk, expectedSize)
		if err != nil {
			if errors.Is(err, backend.ErrNotFound) {
				return fmt.Errorf("%w: source pack %s is missing", ErrCorrupt, pk)
			}
			return fmt.Errorf("vfs: read source pack %s: %w", pk, err)
		}
		if int64(len(data)) != expectedSize {
			return fmt.Errorf("%w: source pack %s length %d, expected %d", ErrCorrupt, pk, len(data), expectedSize)
		}
		_, err = dstCount.PutIfAbsent(wctx, pk, data)
		switch {
		case err == nil:
			packsCopied.Add(1)
			bytesTransferred.Add(int64(len(data)))
			return nil
		case errors.Is(err, backend.ErrConflict):
			sz, serr := dstCount.Stat(wctx, pk)
			if serr != nil {
				return fmt.Errorf("vfs: stat conflicting destination pack %s: %w", pk, serr)
			}
			if sz != int64(len(data)) {
				return fmt.Errorf("%w: destination pack %s conflict has size %d, expected %d", ErrCorrupt, pk, sz, len(data))
			}
			return nil
		default:
			return fmt.Errorf("vfs: write destination pack %s: %w", pk, err)
		}
	}); err != nil {
		return nil, err
	}

	if err := runParallel(ctx, streams, len(plan.ShardsToCopy), &peakG, func(wctx context.Context, idx int) error {
		sk := plan.ShardsToCopy[idx]
		r, ok := ext.shardsToCopyRefs[sk]
		if !ok {
			var perr error
			if r, perr = parseShardRef(sk); perr != nil {
				return perr
			}
		}
		sealed, cached := ext.stagedShards[sk]
		if !cached {
			var err error
			sealed, _, err = srcCount.Get(wctx, sk, maxShard)
			if err != nil {
				if errors.Is(err, backend.ErrNotFound) {
					return fmt.Errorf("%w: source shard %x is missing", ErrCorrupt, r[:6])
				}
				return fmt.Errorf("vfs: read source shard %x: %w", r[:6], err)
			}
		}
		if blake3.Sum(sealed) != r {
			return fmt.Errorf("%w: shard %x does not hash to its name", ErrCorrupt, r[:6])
		}
		if _, err := unseal(c, shardAAD(plan.SrcHead.ID), sealed); err != nil {
			return fmt.Errorf("vfs: open source shard %x: %w", r[:6], err)
		}
		_, err := dstCount.PutIfAbsent(wctx, sk, sealed)
		switch {
		case err == nil:
			shardsCopied.Add(1)
			bytesTransferred.Add(int64(len(sealed)))
			return nil
		case errors.Is(err, backend.ErrConflict):
			sz, serr := dstCount.Stat(wctx, sk)
			if serr != nil {
				return fmt.Errorf("vfs: stat conflicting destination shard %s: %w", sk, serr)
			}
			if sz != int64(len(sealed)) {
				return fmt.Errorf("%w: destination shard %s conflict has size %d, expected %d", ErrCorrupt, sk, sz, len(sealed))
			}
			return nil
		default:
			return fmt.Errorf("vfs: write destination shard %s: %w", sk, err)
		}
	}); err != nil {
		return nil, err
	}
	// Release staged shard ciphertexts before verifying destination.
	ext.stagedShards = nil

	if err := verifyRestoredTree(ctx, dstCount, c, plan.SrcHead, streams, &peakG); err != nil {
		return nil, err
	}

	dstFence := fence.New(dstCount, maxHead)
	if err := dstFence.Rewrite(ctx, headKey, opts.Round, func(payload []byte, recorded uint64) ([]byte, error) {
		if payload != nil {
			if opts.Round <= recorded {
				return nil, fmt.Errorf("%w: round %d does not exceed destination recorded round %d", ErrStaleRound, opts.Round, recorded)
			}
			dstPlain, err := unseal(c, headAAD(dstCount.String()), payload)
			if err != nil {
				return nil, fmt.Errorf("vfs: open destination head: %w", err)
			}
			curHead, err := unmarshalHead(dstPlain)
			if err != nil {
				return nil, fmt.Errorf("vfs: destination head: %w", err)
			}
			if curHead.ID != plan.SrcHead.ID {
				return nil, fmt.Errorf("%w: destination volume id %x does not match source %x", ErrCorrupt, curHead.ID[:6], plan.SrcHead.ID[:6])
			}
			if curHead.Seed != plan.SrcHead.Seed {
				return nil, fmt.Errorf("%w: destination volume seed does not match source", ErrCorrupt)
			}
		}
		return seal(c, headAAD(dstCount.String()), plan.SrcHead.marshal())
	}); err != nil {
		return nil, err
	}

	recordPeakGoroutines(&peakG)
	return &RestoreStats{
		Plan:              *plan,
		ShardsCopied:      int(shardsCopied.Load()),
		PacksCopied:       int(packsCopied.Load()),
		BytesTransferred:  bytesTransferred.Load(),
		FullSnapshotBytes: plan.FullSnapshotBytes,
		SourceGets:        srcCount.gets.Load(),
		DestPuts:          dstCount.puts.Load(),
		DestStats:         dstCount.stats.Load(),
		Duration:          time.Since(startWall),
		CPUTime:           parasyncProcessCPU() - startCPU,
		PeakGoroutines:    int(peakG.Load()),
	}, nil
}

func planDelta(ctx context.Context, src, dst backend.Backend, c Cipher, streams int, retainShards bool, peakG *atomic.Int64) (*DeltaPlan, *planExtras, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	srcFence := fence.New(src, maxHead)
	srcPayload, _, err := srcFence.Get(ctx, headKey)
	if err != nil {
		if errors.Is(err, backend.ErrNotFound) {
			return nil, nil, fmt.Errorf("vfs: no volume at %s: %w", src, fs.ErrNotExist)
		}
		return nil, nil, fmt.Errorf("vfs: read source head: %w", err)
	}
	srcPlain, err := unseal(c, headAAD(src.String()), srcPayload)
	if err != nil {
		return nil, nil, fmt.Errorf("vfs: open source head: %w", err)
	}
	srcHead, err := unmarshalHead(srcPlain)
	if err != nil {
		return nil, nil, fmt.Errorf("vfs: source head: %w", err)
	}

	plan := &DeltaPlan{SrcHead: srcHead}
	ext := &planExtras{
		shardsToCopyRefs: map[string]ref{},
		stagedShards:     map[string][]byte{},
		packSizes:        map[string]int64{},
	}

	dstFence := fence.New(dst, maxHead)
	dstPayload, dstRecorded, err := dstFence.Get(ctx, headKey)
	switch {
	case err == nil:
		dstPlain, uerr := unseal(c, headAAD(dst.String()), dstPayload)
		if uerr != nil {
			return nil, nil, fmt.Errorf("vfs: open destination head: %w", uerr)
		}
		dstHead, herr := unmarshalHead(dstPlain)
		if herr != nil {
			return nil, nil, fmt.Errorf("vfs: destination head: %w", herr)
		}
		if dstHead.ID != srcHead.ID {
			return nil, nil, fmt.Errorf("%w: destination volume id %x does not match source %x", ErrCorrupt, dstHead.ID[:6], srcHead.ID[:6])
		}
		if dstHead.Seed != srcHead.Seed {
			return nil, nil, fmt.Errorf("%w: destination volume seed does not match source", ErrCorrupt)
		}
		plan.HasDst = true
		plan.DstHead = dstHead
		ext.dstRecordedRound = dstRecorded
	case errors.Is(err, backend.ErrNotFound):
		plan.HasDst = false
	default:
		return nil, nil, fmt.Errorf("vfs: read destination head: %w", err)
	}

	var dstNodes map[ref]*dstNodeInfo
	if plan.HasDst && plan.DstHead.Root != none {
		dstNodes, err = collectDstSubtrees(ctx, dst, c, plan.DstHead.ID, plan.DstHead.Root, streams, peakG)
		if err != nil {
			return nil, nil, err
		}
	}

	requiredPacks := map[packID]uint64{}
	if srcHead.Root != none {
		if err := collectSourceDelta(ctx, src, c, srcHead.ID, srcHead.Root, dstNodes, streams, retainShards, peakG, plan, ext, requiredPacks); err != nil {
			return nil, nil, err
		}
	}

	sort.Strings(plan.ShardsToCopy)
	plan.TotalSourceShards = len(plan.ShardsToCopy) + plan.ShardsPruned

	if len(requiredPacks) > 0 {
		type packCheck struct {
			id          packID
			key         string
			requiredEnd uint64
			pruned      bool
			size        int64
		}
		checks := make([]packCheck, 0, len(requiredPacks))
		for id, end := range requiredPacks {
			checks = append(checks, packCheck{
				id:          id,
				key:         packKey(id),
				requiredEnd: end,
			})
		}
		sort.Slice(checks, func(i, j int) bool { return checks[i].key < checks[j].key })

		if err := runParallel(ctx, streams, len(checks), peakG, func(wctx context.Context, idx int) error {
			ch := &checks[idx]
			dstSz, err := dst.Stat(wctx, ch.key)
			if err == nil && uint64(dstSz) >= ch.requiredEnd {
				ch.pruned = true
				ch.size = dstSz
				return nil
			}
			if err != nil && !errors.Is(err, backend.ErrNotFound) {
				return fmt.Errorf("vfs: stat destination pack %s: %w", ch.key, err)
			}
			srcSz, serr := src.Stat(wctx, ch.key)
			if serr != nil {
				if errors.Is(serr, backend.ErrNotFound) {
					return fmt.Errorf("%w: pack %x is missing from source", ErrCorrupt, ch.id[:6])
				}
				return fmt.Errorf("vfs: stat source pack %s: %w", ch.key, serr)
			}
			if uint64(srcSz) < ch.requiredEnd {
				return fmt.Errorf("%w: source pack %x has %d bytes and a chunk ends at %d", ErrCorrupt, ch.id[:6], srcSz, ch.requiredEnd)
			}
			ch.size = srcSz
			return nil
		}); err != nil {
			return nil, nil, err
		}

		for _, ch := range checks {
			plan.FullPackBytes += ch.size
			if ch.pruned {
				plan.PacksPruned++
			} else {
				plan.PacksToCopy = append(plan.PacksToCopy, ch.key)
				plan.PackBytes += ch.size
				ext.packSizes[ch.key] = ch.size
			}
		}
	}

	plan.TotalSourcePacks = len(plan.PacksToCopy) + plan.PacksPruned
	plan.FullSnapshotBytes = plan.FullMetaBytes + plan.FullPackBytes
	return plan, ext, nil
}

func collectDstSubtrees(ctx context.Context, dst backend.Backend, c Cipher, id volumeID, root ref, streams int, peakG *atomic.Int64) (map[ref]*dstNodeInfo, error) {
	aad := shardAAD(id)
	nodes := map[ref]*dstNodeInfo{}
	var levels [][]ref
	frontier := []ref{root}
	expectedLevel := -1

	for len(frontier) > 0 {
		type fetchRes struct {
			r    ref
			info *dstNodeInfo
		}
		res := make([]fetchRes, len(frontier))
		if err := runParallel(ctx, streams, len(frontier), peakG, func(wctx context.Context, idx int) error {
			r := frontier[idx]
			sealed, _, err := dst.Get(wctx, shardKey(r), maxShard)
			if err != nil {
				if errors.Is(err, backend.ErrNotFound) {
					return fmt.Errorf("%w: destination shard %x is missing", ErrCorrupt, r[:6])
				}
				return fmt.Errorf("vfs: read destination shard %x: %w", r[:6], err)
			}
			if blake3.Sum(sealed) != r {
				return fmt.Errorf("%w: destination shard %x does not hash to its name", ErrCorrupt, r[:6])
			}
			plain, err := unseal(c, aad, sealed)
			if err != nil {
				return fmt.Errorf("vfs: open destination shard %x: %w", r[:6], err)
			}
			n, err := decodeNode(plain)
			if err != nil {
				return fmt.Errorf("vfs: decode destination shard %x: %w", r[:6], err)
			}
			if expectedLevel >= 0 && n.level != expectedLevel {
				return fmt.Errorf("%w: destination shard %x has level %d, want %d", ErrCorrupt, r[:6], n.level, expectedLevel)
			}
			info := &dstNodeInfo{
				level:     n.level,
				sealedLen: int64(len(sealed)),
			}
			if n.level > 0 {
				info.children = make([]ref, len(n.vals))
				for i, v := range n.vals {
					info.children[i] = ref(v)
				}
			} else {
				packs, err := extractLeafPacks(n)
				if err != nil {
					return err
				}
				info.packs = packs
			}
			res[idx] = fetchRes{r: r, info: info}
			return nil
		}); err != nil {
			return nil, err
		}

		levelRefs := make([]ref, 0, len(res))
		var nextFrontier []ref
		levelVal := -1
		for _, item := range res {
			nodes[item.r] = item.info
			levelRefs = append(levelRefs, item.r)
			if levelVal < 0 {
				levelVal = item.info.level
			} else if item.info.level != levelVal {
				return nil, fmt.Errorf("%w: destination tree level mismatch (%d vs %d)", ErrCorrupt, item.info.level, levelVal)
			}
			for _, ch := range item.info.children {
				if _, seen := nodes[ch]; !seen {
					nodes[ch] = nil // reserve against duplicates in same tree
					nextFrontier = append(nextFrontier, ch)
				}
			}
		}
		levels = append(levels, levelRefs)
		if levelVal == 0 && len(nextFrontier) > 0 {
			return nil, fmt.Errorf("%w: leaf shard has children", ErrCorrupt)
		}
		expectedLevel = levelVal - 1
		frontier = dedupRefs(nextFrontier)
	}

	// Aggregate exact subtree shard counts and sealed byte totals bottom-up.
	for i := len(levels) - 1; i >= 0; i-- {
		for _, r := range levels[i] {
			info := nodes[r]
			shards := 1
			metaBytes := info.sealedLen
			for _, ch := range info.children {
				cinfo := nodes[ch]
				if cinfo == nil {
					return nil, fmt.Errorf("%w: destination subtree %x missing child %x", ErrCorrupt, r[:6], ch[:6])
				}
				shards += cinfo.subtreeShards
				metaBytes += cinfo.subtreeMetaBytes
			}
			info.subtreeShards = shards
			info.subtreeMetaBytes = metaBytes
		}
	}

	return nodes, nil
}

func collectSourceDelta(
	ctx context.Context,
	src backend.Backend,
	c Cipher,
	id volumeID,
	root ref,
	dstNodes map[ref]*dstNodeInfo,
	streams int,
	retainShards bool,
	peakG *atomic.Int64,
	plan *DeltaPlan,
	ext *planExtras,
	requiredPacks map[packID]uint64,
) error {
	aad := shardAAD(id)
	visited := map[ref]struct{}{}
	var stagedBytes atomic.Int64

	type srcItem struct {
		r             ref
		expectedLevel int // -1 for root
	}

	pruneSubtree := func(r ref, expectedLevel int) (bool, error) {
		if dstNodes == nil {
			return false, nil
		}
		info, ok := dstNodes[r]
		if !ok || info == nil {
			return false, nil
		}
		if expectedLevel >= 0 && info.level != expectedLevel {
			return false, fmt.Errorf("%w: source subtree %x level %d mismatches destination level %d", ErrCorrupt, r[:6], expectedLevel, info.level)
		}
		plan.ShardsPruned += info.subtreeShards
		plan.FullMetaBytes += info.subtreeMetaBytes
		collectPrunedPacks(dstNodes, r, requiredPacks)
		return true, nil
	}

	visited[root] = struct{}{}
	if pruned, err := pruneSubtree(root, -1); err != nil {
		return err
	} else if pruned {
		return nil
	}

	frontier := []srcItem{{r: root, expectedLevel: -1}}
	for len(frontier) > 0 {
		type shardRes struct {
			r         ref
			key       string
			level     int
			sealedLen int64
			sealed    []byte
			children  []ref
			packs     map[packID]uint64
		}
		results := make([]shardRes, len(frontier))
		if err := runParallel(ctx, streams, len(frontier), peakG, func(wctx context.Context, idx int) error {
			it := frontier[idx]
			sk := shardKey(it.r)
			sealed, _, err := src.Get(wctx, sk, maxShard)
			if err != nil {
				if errors.Is(err, backend.ErrNotFound) {
					return fmt.Errorf("%w: source shard %x is missing", ErrCorrupt, it.r[:6])
				}
				return fmt.Errorf("vfs: read source shard %x: %w", it.r[:6], err)
			}
			if blake3.Sum(sealed) != it.r {
				return fmt.Errorf("%w: shard %x does not hash to its name", ErrCorrupt, it.r[:6])
			}
			plain, err := unseal(c, aad, sealed)
			if err != nil {
				return fmt.Errorf("vfs: open shard %x: %w", it.r[:6], err)
			}
			n, err := decodeNode(plain)
			if err != nil {
				return fmt.Errorf("vfs: decode shard %x: %w", it.r[:6], err)
			}
			if it.expectedLevel >= 0 && n.level != it.expectedLevel {
				return fmt.Errorf("%w: source shard %x has level %d, want %d", ErrCorrupt, it.r[:6], n.level, it.expectedLevel)
			}

			out := shardRes{
				r:         it.r,
				key:       sk,
				level:     n.level,
				sealedLen: int64(len(sealed)),
			}
			if retainShards && stagedBytes.Add(int64(len(sealed))) <= maxParaSyncStagedShardBytes {
				out.sealed = sealed
			}
			if n.level > 0 {
				out.children = make([]ref, len(n.vals))
				for i, v := range n.vals {
					out.children[i] = ref(v)
				}
			} else {
				packs, err := extractLeafPacks(n)
				if err != nil {
					return err
				}
				out.packs = packs
			}
			results[idx] = out
			return nil
		}); err != nil {
			return err
		}

		var nextFrontier []srcItem
		for _, res := range results {
			plan.ShardsToCopy = append(plan.ShardsToCopy, res.key)
			plan.MetaBytes += res.sealedLen
			plan.FullMetaBytes += res.sealedLen
			ext.shardsToCopyRefs[res.key] = res.r
			if res.sealed != nil {
				ext.stagedShards[res.key] = res.sealed
			}
			for id, end := range res.packs {
				requiredPacks[id] = max(requiredPacks[id], end)
			}
			if res.level > 0 {
				childLevel := res.level - 1
				for _, ch := range res.children {
					if _, seen := visited[ch]; seen {
						continue
					}
					visited[ch] = struct{}{}
					pruned, err := pruneSubtree(ch, childLevel)
					if err != nil {
						return err
					}
					if !pruned {
						nextFrontier = append(nextFrontier, srcItem{r: ch, expectedLevel: childLevel})
					}
				}
			}
		}
		frontier = nextFrontier
	}
	return nil
}

func collectPrunedPacks(dstNodes map[ref]*dstNodeInfo, root ref, out map[packID]uint64) {
	stack := []ref{root}
	for len(stack) > 0 {
		r := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		info := dstNodes[r]
		if info == nil {
			continue
		}
		if info.level == 0 {
			for id, end := range info.packs {
				out[id] = max(out[id], end)
			}
			continue
		}
		stack = append(stack, info.children...)
	}
}

func extractLeafPacks(n *node) (map[packID]uint64, error) {
	var packs map[packID]uint64
	for i, k := range n.keys {
		if len(k) == 0 || k[0] != tChunk {
			continue
		}
		if len(k) != 1+len(chunkID{}) {
			return nil, fmt.Errorf("%w: chunk table key of %d bytes", ErrCorrupt, len(k))
		}
		w, err := decodeWhere(n.vals[i])
		if err != nil {
			return nil, err
		}
		if w.off+w.n < w.off {
			return nil, fmt.Errorf("%w: chunk extent overflow", ErrCorrupt)
		}
		if packs == nil {
			packs = map[packID]uint64{}
		}
		packs[w.pack] = max(packs[w.pack], w.off+w.n)
	}
	return packs, nil
}

// verifyRestoredTree warms verifyVol's shard cache in parallel from dst and runs
// checkTree(ctx, true) to enforce every canonical tree and pack invariant before
// updating destination head.
func verifyRestoredTree(ctx context.Context, dst backend.Backend, c Cipher, head Head, streams int, peakG *atomic.Int64) error {
	vCfg, err := Config{Backend: dst, Cipher: c, Streams: streams}.valid()
	if err != nil {
		return err
	}
	verifyVol := newVolume(vCfg)
	defer verifyVol.stop()
	verifyVol.adopt(head)

	if head.Root != none {
		visited := map[ref]struct{}{head.Root: {}}
		frontier := []ref{head.Root}
		expectedLevel := -1
		for len(frontier) > 0 {
			nodes := make([]*node, len(frontier))
			if err := runParallel(ctx, streams, len(frontier), peakG, func(wctx context.Context, idx int) error {
				n, err := verifyVol.t.sh.load(wctx, frontier[idx])
				if err != nil {
					return err
				}
				if expectedLevel >= 0 && n.level != expectedLevel {
					return fmt.Errorf("%w: shard %x has level %d, want %d", ErrCorrupt, frontier[idx][:6], n.level, expectedLevel)
				}
				nodes[idx] = n
				return nil
			}); err != nil {
				return fmt.Errorf("vfs: verify destination shards: %w", err)
			}
			var next []ref
			levelVal := -1
			for _, n := range nodes {
				if levelVal < 0 {
					levelVal = n.level
				} else if n.level != levelVal {
					return fmt.Errorf("%w: destination tree level mismatch (%d vs %d)", ErrCorrupt, n.level, levelVal)
				}
				if n.level > 0 {
					for _, v := range n.vals {
						ch := ref(v)
						if _, ok := visited[ch]; !ok {
							visited[ch] = struct{}{}
							next = append(next, ch)
						}
					}
				}
			}
			expectedLevel = levelVal - 1
			frontier = next
		}
	}
	if err := verifyVol.ns.load(head); err != nil {
		return fmt.Errorf("vfs: verify head metadata: %w", err)
	}
	return verifyVol.checkTree(ctx, true)
}

func runParallel(ctx context.Context, workers, n int, peakG *atomic.Int64, fn func(context.Context, int) error) error {
	if n <= 0 {
		recordPeakGoroutines(peakG)
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	w := min(max(workers, 1), n)
	if w == 1 {
		recordPeakGoroutines(peakG)
		for i := range n {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := fn(ctx, i); err != nil {
				return err
			}
		}
		return nil
	}

	wctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		next     atomic.Int64
		errMu    sync.Mutex
		firstErr error
		wg       sync.WaitGroup
	)
	setErr := func(err error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
			cancel()
		}
		errMu.Unlock()
	}

	for range w {
		wg.Add(1)
		go func() {
			defer wg.Done()
			recordPeakGoroutines(peakG)
			for {
				if wctx.Err() != nil {
					return
				}
				idx := int(next.Add(1) - 1)
				if idx >= n {
					return
				}
				if err := fn(wctx, idx); err != nil {
					setErr(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	recordPeakGoroutines(peakG)
	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}

func recordPeakGoroutines(peak *atomic.Int64) {
	if peak == nil {
		return
	}
	cur := int64(runtime.NumGoroutine())
	for {
		prev := peak.Load()
		if cur <= prev || peak.CompareAndSwap(prev, cur) {
			return
		}
	}
}

func dedupRefs(in []ref) []ref {
	if len(in) <= 1 {
		return in
	}
	seen := make(map[ref]struct{}, len(in))
	out := in[:0]
	for _, r := range in {
		if _, ok := seen[r]; !ok {
			seen[r] = struct{}{}
			out = append(out, r)
		}
	}
	return out
}

func parseShardRef(key string) (ref, error) {
	const prefix = "meta/"
	if !strings.HasPrefix(key, prefix) {
		return none, fmt.Errorf("%w: invalid shard key %q", ErrCorrupt, key)
	}
	rest := key[len(prefix):]
	dir, hexRef, ok := strings.Cut(rest, "/")
	if !ok || len(dir) != 2 || len(hexRef) != 64 || hexRef[:2] != dir {
		return none, fmt.Errorf("%w: invalid shard key %q", ErrCorrupt, key)
	}
	b, err := hex.DecodeString(hexRef)
	if err != nil || len(b) != len(ref{}) {
		return none, fmt.Errorf("%w: invalid shard key %q", ErrCorrupt, key)
	}
	var r ref
	copy(r[:], b)
	return r, nil
}

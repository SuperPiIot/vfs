// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package dogi

import (
	"context"
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/luxfi/age"

	"github.com/hanzoai/vfs"
	"github.com/hanzoai/vfs/pkg/backend"
	_ "github.com/hanzoai/vfs/pkg/backend/file"
)

// ProbeBackend wraps a backend.Backend to count every real object-store call
// and optionally inject an explicitly labeled simulated per-request delay.
type ProbeBackend struct {
	backend.Backend
	SimDelay time.Duration

	Gets       atomic.Int64
	GetRanges  atomic.Int64
	Puts       atomic.Int64
	PutAbsents atomic.Int64
	PutVers    atomic.Int64
	Deletes    atomic.Int64
	Stats      atomic.Int64
}

// NewProbeBackend wraps be with real request counting and optional simulated delay.
func NewProbeBackend(be backend.Backend, simDelay time.Duration) *ProbeBackend {
	return &ProbeBackend{Backend: be, SimDelay: simDelay}
}

func (p *ProbeBackend) delay() {
	if p.SimDelay > 0 {
		time.Sleep(p.SimDelay)
	}
}

func (p *ProbeBackend) Get(ctx context.Context, key string, max int64) ([]byte, backend.Version, error) {
	p.delay()
	p.Gets.Add(1)
	return p.Backend.Get(ctx, key, max)
}

func (p *ProbeBackend) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	p.delay()
	p.GetRanges.Add(1)
	return p.Backend.GetRange(ctx, key, off, n)
}

func (p *ProbeBackend) Put(ctx context.Context, key string, data []byte) error {
	p.delay()
	p.Puts.Add(1)
	return p.Backend.Put(ctx, key, data)
}

func (p *ProbeBackend) PutIfAbsent(ctx context.Context, key string, data []byte) (backend.Version, error) {
	p.delay()
	p.PutAbsents.Add(1)
	return p.Backend.PutIfAbsent(ctx, key, data)
}

func (p *ProbeBackend) PutIfVersion(ctx context.Context, key string, data []byte, v backend.Version) (backend.Version, error) {
	p.delay()
	p.PutVers.Add(1)
	return p.Backend.PutIfVersion(ctx, key, data, v)
}

func (p *ProbeBackend) Delete(ctx context.Context, key string) error {
	p.delay()
	p.Deletes.Add(1)
	return p.Backend.Delete(ctx, key)
}

func (p *ProbeBackend) Stat(ctx context.Context, key string) (int64, error) {
	p.delay()
	p.Stats.Add(1)
	return p.Backend.Stat(ctx, key)
}

// TotalRequests returns the sum of all tracked backend object operations.
func (p *ProbeBackend) TotalRequests() int64 {
	return p.Gets.Load() + p.GetRanges.Load() + p.Puts.Load() +
		p.PutAbsents.Load() + p.PutVers.Load() + p.Deletes.Load() + p.Stats.Load()
}

// TraceOp represents a single deterministic operation in a synthetic coding-agent
// build and workspace trace.
type TraceOp struct {
	Round        int
	Kind         string // "put", "delete", "checkpoint", "compact"
	Key          string
	Meta         ObjectMeta
	TrueLifetime Lifetime
	TrueTTL      time.Duration
	Payload      []byte
}

// AgentTrace holds a chronologically split 40% training / 60% held-out workload
// plus a 100%-long-lived zero-expiration negative control trace.
type AgentTrace struct {
	TrainRecords      []TrainingRecord
	HeldOutRecords    []TrainingRecord
	HeldOutOps        []TraceOp
	ZeroExpirationOps []TraceOp
	OracleTruth       map[string]Lifetime
}

// GenerateAgentTrace constructs a deterministic coding-agent build + workspace
// workload split chronologically 40% train / 60% held-out. The held-out split
// uses disjoint workspace prefixes, unseen file names, overlapping extensions
// (such as .bin and .log), and organic ~4% TTL variance so the learned
// classifier must generalize without memorization or oracle cheating.
func GenerateAgentTrace(seed uint64, totalRounds, filesPerRound, avgChunkSize int) AgentTrace {
	if totalRounds < 5 {
		totalRounds = 10
	}
	if filesPerRound < 10 {
		filesPerRound = 30
	}
	if avgChunkSize < 1024 {
		avgChunkSize = 16 << 10
	}
	trainRounds := max(2, (totalRounds*4)/10)
	rng := rand.New(rand.NewPCG(seed, seed^0x517cc1b727220a95))

	trainWorkspaces := []string{"ws-alpha", "ws-beta"}
	evalWorkspaces := []string{"ws-gamma", "ws-delta", "ws-epsilon"}

	var (
		trainRecs   []TrainingRecord
		heldOutRecs []TrainingRecord
		heldOutOps  []TraceOp
		zeroExpOps  []TraceOp
		oracleTruth = make(map[string]Lifetime)
	)

	// Scheduled deletions by round for held-out simulation.
	expireAt := make(map[int][]string)
	fileSeq := 0

	makePayload := func(size int, tag uint64) []byte {
		b := make([]byte, size)
		r := rand.New(rand.NewPCG(tag, seed+17))
		for i := 0; i < len(b); i += 8 {
			v := r.Uint64()
			for j := 0; j < 8 && i+j < len(b); j++ {
				b[i+j] = byte(v >> (8 * j))
			}
		}
		return b
	}

	sampleObject := func(round int, ws string, seq int) (key string, meta ObjectMeta, cls Lifetime, ttl time.Duration, sz int) {
		roll := rng.Float64()
		jitter := rng.Float64()
		switch {
		case roll < 0.58:
			// Ephemeral build temporaries (~58% of writes): .o, .tmp, .d, /target/, /node_modules/.cache/
			cls = LifetimeEphemeral
			ttl = time.Duration(30+rng.IntN(90)) * time.Second
			sub := seq % 4
			sz = avgChunkSize/2 + rng.IntN(avgChunkSize)
			switch sub {
			case 0:
				key = fmt.Sprintf("/%s/target/debug/deps/crate_%04d.o", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.72 + rng.Float64()*0.25, ProcessTag: "rustc"}
			case 1:
				key = fmt.Sprintf("/%s/node_modules/.cache/babel/chunk_%04d.tmp", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.70 + rng.Float64()*0.25, ProcessTag: "npm"}
			case 2:
				key = fmt.Sprintf("/%s/tmp/build_r%02d/unit_%04d.d", ws, round, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.78 + rng.Float64()*0.20, ProcessTag: "go"}
			default:
				key = fmt.Sprintf("/%s/tmp/scratch/run_%04d.log", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.75 + rng.Float64()*0.20, ProcessTag: "agent"}
			}
			// 4% organic noise: occasional temp artifact lingers for 2 rounds.
			if jitter < 0.04 {
				cls = LifetimeMedium
				ttl = 15 * time.Minute
			}

		case roll < 0.80:
			// Medium-lived build intermediates & artifacts (~22% of writes)
			cls = LifetimeMedium
			ttl = time.Duration(10+rng.IntN(25)) * time.Minute
			sub := seq % 3
			sz = avgChunkSize + rng.IntN(avgChunkSize)
			switch sub {
			case 0:
				key = fmt.Sprintf("/%s/dist/bundle_%04d.tar", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.30 + rng.Float64()*0.20, ProcessTag: "ci"}
			case 1:
				// Overlapping .bin extension shared with model weights (different dir + overwrite rate).
				key = fmt.Sprintf("/%s/artifacts/stage_%04d.bin", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.32 + rng.Float64()*0.25, ProcessTag: "ci"}
			default:
				key = fmt.Sprintf("/%s/build/reports/suite_%04d.log", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: 0.28 + rng.Float64()*0.22, ProcessTag: "go"}
			}
			if jitter < 0.04 {
				cls = LifetimeEphemeral
				ttl = 60 * time.Second
			}

		default:
			// Long-lived source code, model weights, checkpoints (~20% of writes)
			cls = LifetimeLong
			ttl = 24 * time.Hour
			sub := seq % 4
			sz = avgChunkSize + rng.IntN(avgChunkSize*3/2)
			switch sub {
			case 0:
				key = fmt.Sprintf("/%s/src/pkg_%02d/module_%04d.go", ws, seq%16, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: rng.Float64() * 0.04, ProcessTag: "editor"}
			case 1:
				key = fmt.Sprintf("/%s/src/core/engine_%04d.rs", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: rng.Float64() * 0.04, ProcessTag: "editor"}
			case 2:
				key = fmt.Sprintf("/%s/checkpoints/step_%02d/weights_%04d.safetensors", ws, round, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: rng.Float64() * 0.03, ProcessTag: "ckpt"}
			default:
				// Overlapping .bin extension in models/ with near-zero overwrite rate.
				key = fmt.Sprintf("/%s/models/embed_%04d.bin", ws, seq)
				meta = ObjectMeta{Path: key, Size: int64(sz), OverwriteRate: rng.Float64() * 0.03, ProcessTag: "ckpt"}
			}
			if jitter < 0.03 {
				cls = LifetimeMedium
				ttl = 20 * time.Minute
			}
		}
		return key, meta, cls, ttl, sz
	}

	for r := range totalRounds {
		isTrain := r < trainRounds
		wsPool := evalWorkspaces
		if isTrain {
			wsPool = trainWorkspaces
		}

		for i := range filesPerRound {
			fileSeq++
			ws := wsPool[(r+i)%len(wsPool)]
			key, meta, cls, ttl, sz := sampleObject(r, ws, fileSeq)
			rec := TrainingRecord{
				Meta:         meta,
				TrueLifetime: cls,
				TrueTTL:      ttl,
			}
			if isTrain {
				trainRecs = append(trainRecs, rec)
				continue
			}

			heldOutRecs = append(heldOutRecs, rec)
			oracleTruth[key] = cls
			payload := makePayload(sz, uint64(fileSeq))
			heldOutOps = append(heldOutOps, TraceOp{
				Round:        r,
				Kind:         "put",
				Key:          key,
				Meta:         meta,
				TrueLifetime: cls,
				TrueTTL:      ttl,
				Payload:      payload,
			})

			// Zero-expiration boundary control: mixed lifetime classes (all 3 streams active)
			// with zero expirations/deletions (e.g., retained build archive snapshot).
			zeroKey := fmt.Sprintf("/%s/retained%s", ws, key[len("/"+ws):])
			zeroMeta := meta
			zeroMeta.Path = zeroKey
			zeroExpOps = append(zeroExpOps, TraceOp{
				Round:        r,
				Kind:         "put",
				Key:          zeroKey,
				Meta:         zeroMeta,
				TrueLifetime: cls,
				TrueTTL:      24 * time.Hour,
				Payload:      payload,
			})

			switch cls {
			case LifetimeEphemeral:
				expireAt[r] = append(expireAt[r], key)
			case LifetimeMedium:
				expireAt[r+2] = append(expireAt[r+2], key)
			}
		}

		if !isTrain {
			heldOutOps = append(heldOutOps, TraceOp{Round: r, Kind: "checkpoint"})
			zeroExpOps = append(zeroExpOps, TraceOp{Round: r, Kind: "checkpoint"})
			for _, delKey := range expireAt[r] {
				heldOutOps = append(heldOutOps, TraceOp{Round: r, Kind: "delete", Key: delKey})
			}
			heldOutOps = append(heldOutOps, TraceOp{Round: r, Kind: "compact"})
		}
	}

	// Final cleanup round for remaining medium files from penultimate rounds.
	lastR := totalRounds
	if len(expireAt[lastR]) > 0 || len(expireAt[lastR+1]) > 0 {
		for _, k := range expireAt[lastR] {
			heldOutOps = append(heldOutOps, TraceOp{Round: lastR, Kind: "delete", Key: k})
		}
		for _, k := range expireAt[lastR+1] {
			heldOutOps = append(heldOutOps, TraceOp{Round: lastR, Kind: "delete", Key: k})
		}
		heldOutOps = append(heldOutOps, TraceOp{Round: lastR, Kind: "compact"})
	}

	return AgentTrace{
		TrainRecords:      trainRecs,
		HeldOutRecords:    heldOutRecs,
		HeldOutOps:        heldOutOps,
		ZeroExpirationOps: zeroExpOps,
		OracleTruth:       oracleTruth,
	}
}

// ModeMetrics records real measured storage, request, timing, and memory
// telemetry for a single unified Packer evaluation run.
type ModeMetrics struct {
	Scenario               string  `json:"scenario"`
	Name                   string  `json:"name"`
	Mode                   string  `json:"mode"`
	NumStreams             int     `json:"num_streams"`
	PackTargetBytes        int     `json:"pack_target_bytes"`
	GarbageRatio           float64 `json:"garbage_ratio"`
	ClassifierAccuracy     float64 `json:"classifier_accuracy"`
	SevereMispredCount     int     `json:"severe_mispred_count"`
	LogicalChunks          int64   `json:"logical_chunks"`
	LogicalPlaintextBytes  int64   `json:"logical_plaintext_bytes"`
	LogicalSealedBytes     int64   `json:"logical_sealed_bytes"`
	PhysicalBytesWritten   int64   `json:"physical_bytes_written"`
	PhysicalBytesReclaimed int64   `json:"physical_bytes_reclaimed"`
	ActivePhysicalBytes    int64   `json:"active_physical_bytes"`
	LiveBytesCopied        int64   `json:"live_bytes_copied"`
	WriteAmplification     float64 `json:"write_amplification"`
	PacksCreated           int64   `json:"packs_created"`
	InitialPacksCreated    int64   `json:"initial_packs_created"`
	CompactionPacksCreated int64   `json:"compaction_packs_created"`
	PacksDeleted           int64   `json:"packs_deleted"`
	PackCountOverheadPct   float64 `json:"pack_count_overhead_pct"`
	LiveBytesReductionPct  float64 `json:"live_bytes_reduction_pct"`
	WAFReductionPct        float64 `json:"waf_reduction_pct"`
	PutRequests            int64   `json:"put_requests"`
	GetRangeRequests       int64   `json:"get_range_requests"`
	DeleteRequests         int64   `json:"delete_requests"`
	TotalBackendRequests   int64   `json:"total_backend_requests"`
	WallMs                 float64 `json:"wall_ms"`
	SimulatedRTTMs         float64 `json:"simulated_rtt_ms"`
	HeapAllocDeltaBytes    uint64  `json:"heap_alloc_delta_bytes"`
}

// SuiteConfig configures RunSuite.
type SuiteConfig struct {
	ScratchDir   string
	FullScale    bool
	SimulatedRTT time.Duration // optional injected simulation delay (0 = unthrottled real run)
	Seed         uint64
}

// SuiteReport bundles all measured comparisons across baseline, learned DOGI,
// oracle upper bound, adversarial misprediction sweeps, PackTarget sweeps,
// and negative/boundary controls.
type SuiteReport struct {
	Scale              string        `json:"scale"`
	TrainSamples       int           `json:"train_samples"`
	HeldOutSamples     int           `json:"held_out_samples"`
	HeldOutEval        EvalReport    `json:"held_out_eval"`
	PrimaryRuns        []ModeMetrics `json:"primary_runs"`
	AdversarialSweep   []ModeMetrics `json:"adversarial_sweep"`
	PackTargetSweep    []ModeMetrics `json:"pack_target_sweep"`
	BoundaryControls   []ModeMetrics `json:"boundary_controls"`
	SimulatedRTTChecks []ModeMetrics `json:"simulated_rtt_checks,omitempty"`
}

// RunSuite executes all DOGI comparative benchmarks using the unified Packer,
// real vfs.Hybrid post-quantum Cipher, and real file:// backend stores.
func RunSuite(ctx context.Context, cfg SuiteConfig) (SuiteReport, error) {
	if cfg.Seed == 0 {
		cfg.Seed = 0x20261344
	}
	scratch := cfg.ScratchDir
	if scratch == "" {
		var err error
		scratch, err = os.MkdirTemp("", "dogi-bench-*")
		if err != nil {
			return SuiteReport{}, err
		}
		defer os.RemoveAll(scratch)
	}

	rounds, filesPerRound, avgChunk := 10, 24, 16<<10
	primaryTarget := 64 << 10
	scaleName := "standard"
	if cfg.FullScale {
		rounds, filesPerRound, avgChunk = 12, 40, 24<<10
		primaryTarget = 128 << 10
		scaleName = "full"
	}

	trace := GenerateAgentTrace(cfg.Seed, rounds, filesPerRound, avgChunk)

	// Train strictly on the 40% training split; never expose HeldOutRecords to Train.
	clf := NewClassifier()
	clf.Train(trace.TrainRecords)
	eval := EvaluateClassifier(clf, trace.HeldOutRecords)
	oracle := &OraclePredictor{Truth: trace.OracleTruth}

	key, err := age.GenerateHybridIdentity()
	if err != nil {
		return SuiteReport{}, err
	}
	ciph := vfs.Hybrid(key)

	runOne := func(scenario, name string, mode Mode, pred Predictor, packTarget int, ops []TraceOp, simDelay time.Duration) (ModeMetrics, error) {
		dir := filepath.Join(scratch, strings.ReplaceAll(scenario+"_"+name, " ", "_"))
		_ = os.RemoveAll(dir)
		_ = os.MkdirAll(dir, 0o755)
		rawBE, err := backend.Open(ctx, "file://"+filepath.ToSlash(dir))
		if err != nil {
			return ModeMetrics{}, err
		}
		defer rawBE.Close()
		probe := NewProbeBackend(rawBE, simDelay)

		pk, err := NewPacker(ctx, Config{
			Backend:    probe,
			Cipher:     ciph,
			Round:      1,
			Mode:       mode,
			PackTarget: packTarget,
			Predictor:  pred,
		})
		if err != nil {
			return ModeMetrics{}, err
		}

		var mBefore, mAfter runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&mBefore)
		t0 := time.Now()

		var compactPacksCreated int64
		for _, op := range ops {
			switch op.Kind {
			case "put":
				if err := pk.Put(ctx, op.Key, op.Payload, op.Meta); err != nil {
					return ModeMetrics{}, err
				}
			case "delete":
				if err := pk.Delete(op.Key); err != nil {
					return ModeMetrics{}, err
				}
			case "checkpoint":
				if err := pk.Checkpoint(ctx); err != nil {
					return ModeMetrics{}, err
				}
			case "compact":
				beforePacks := pk.Stats().PacksCreated
				if _, err := pk.Compact(ctx, CompactionPolicy{GarbageRatio: 0.40}); err != nil {
					return ModeMetrics{}, err
				}
				compactPacksCreated += pk.Stats().PacksCreated - beforePacks
			}
		}
		if err := pk.Checkpoint(ctx); err != nil {
			return ModeMetrics{}, err
		}
		if err := pk.CheckAccounting(); err != nil {
			return ModeMetrics{}, err
		}
		// Spot-check surviving live keys can be opened and verified cleanly.
		liveKeys := pk.LiveKeys()
		for i := 0; i < len(liveKeys) && i < 12; i++ {
			if _, err := pk.Get(ctx, liveKeys[i]); err != nil {
				return ModeMetrics{}, fmt.Errorf("verify live key %s: %w", liveKeys[i], err)
			}
		}

		wallMs := float64(time.Since(t0).Microseconds()) / 1000.0
		runtime.ReadMemStats(&mAfter)
		var heapDelta uint64
		if mAfter.HeapAlloc > mBefore.HeapAlloc {
			heapDelta = mAfter.HeapAlloc - mBefore.HeapAlloc
		}

		st := pk.Stats()
		acc := 1.0
		severe := 0
		if pred != nil {
			ev := EvaluateClassifier(pred, trace.HeldOutRecords)
			acc = ev.Accuracy
			severe = ev.SevereMispredictions
		} else {
			acc = 0
		}

		return ModeMetrics{
			Scenario:               scenario,
			Name:                   name,
			Mode:                   mode.String(),
			NumStreams:             mode.NumStreams(),
			PackTargetBytes:        packTarget,
			GarbageRatio:           0.40,
			ClassifierAccuracy:     acc,
			SevereMispredCount:     severe,
			LogicalChunks:          st.LogicalChunksWritten,
			LogicalPlaintextBytes:  st.LogicalPlaintextBytes,
			LogicalSealedBytes:     st.LogicalSealedBytes,
			PhysicalBytesWritten:   st.PhysicalBytesWritten,
			PhysicalBytesReclaimed: st.PhysicalBytesReclaimed,
			ActivePhysicalBytes:    st.ActivePhysicalBytes(),
			LiveBytesCopied:        st.LiveBytesCopied,
			WriteAmplification:     st.WriteAmplification(),
			PacksCreated:           st.PacksCreated,
			InitialPacksCreated:    st.PacksCreated - compactPacksCreated,
			CompactionPacksCreated: compactPacksCreated,
			PacksDeleted:           st.PacksDeleted,
			PutRequests:            probe.PutAbsents.Load() + probe.PutVers.Load() + probe.Puts.Load(),
			GetRangeRequests:       probe.GetRanges.Load(),
			DeleteRequests:         probe.Deletes.Load(),
			TotalBackendRequests:   probe.TotalRequests(),
			WallMs:                 wallMs,
			SimulatedRTTMs:         float64(simDelay.Microseconds()) / 1000.0,
			HeapAllocDeltaBytes:    heapDelta,
		}, nil
	}

	annotateRelative := func(base ModeMetrics, m ModeMetrics) ModeMetrics {
		if base.PacksCreated > 0 {
			m.PackCountOverheadPct = (float64(m.PacksCreated-base.PacksCreated) / float64(base.PacksCreated)) * 100.0
		}
		if base.LiveBytesCopied > 0 {
			m.LiveBytesReductionPct = (1.0 - float64(m.LiveBytesCopied)/float64(base.LiveBytesCopied)) * 100.0
		}
		if base.WriteAmplification > 0 {
			m.WAFReductionPct = (1.0 - m.WriteAmplification/base.WriteAmplification) * 100.0
		}
		return m
	}

	// 1. Primary comparison (unthrottled real file:// backend + vfs.Hybrid).
	baseRes, err := runOne("primary", "baseline-fifo", ModeBaselineSingleStream, nil, primaryTarget, trace.HeldOutOps, 0)
	if err != nil {
		return SuiteReport{}, err
	}
	learnedRes, err := runOne("primary", "dogi-learned", ModeLifetimeSeparated, clf, primaryTarget, trace.HeldOutOps, 0)
	if err != nil {
		return SuiteReport{}, err
	}
	oracleRes, err := runOne("primary", "dogi-oracle", ModeLifetimeSeparated, oracle, primaryTarget, trace.HeldOutOps, 0)
	if err != nil {
		return SuiteReport{}, err
	}
	learnedRes = annotateRelative(baseRes, learnedRes)
	oracleRes = annotateRelative(baseRes, oracleRes)

	// 2. Adversarial misprediction sweep (injected flip rate over trained classifier).
	var advSweep []ModeMetrics
	for _, flip := range []float64{0.10, 0.25, 0.45, 0.75} {
		advPred := NewAdversarialPredictor(clf, flip, cfg.Seed+uint64(flip*1000))
		lbl := fmt.Sprintf("dogi-noise-%dpct", int(flip*100))
		m, err := runOne("adversarial", lbl, ModeLifetimeSeparated, advPred, primaryTarget, trace.HeldOutOps, 0)
		if err != nil {
			return SuiteReport{}, err
		}
		advSweep = append(advSweep, annotateRelative(baseRes, m))
	}

	// 3. PackTarget sensitivity sweep (64 KiB, 256 KiB).
	var ptSweep []ModeMetrics
	for _, pt := range []int{32 << 10, 64 << 10, 256 << 10} {
		bPT, err := runOne("packtarget", fmt.Sprintf("baseline-%dk", pt>>10), ModeBaselineSingleStream, nil, pt, trace.HeldOutOps, 0)
		if err != nil {
			return SuiteReport{}, err
		}
		dPT, err := runOne("packtarget", fmt.Sprintf("dogi-%dk", pt>>10), ModeLifetimeSeparated, clf, pt, trace.HeldOutOps, 0)
		if err != nil {
			return SuiteReport{}, err
		}
		ptSweep = append(ptSweep, bPT, annotateRelative(bPT, dPT))
	}

	// 4. Negative / Boundary control: 100% long-lived zero-expiration workload.
	ctrlBase, err := runOne("boundary-no-expiry", "baseline-fifo", ModeBaselineSingleStream, nil, primaryTarget, trace.ZeroExpirationOps, 0)
	if err != nil {
		return SuiteReport{}, err
	}
	ctrlDogi, err := runOne("boundary-no-expiry", "dogi-learned", ModeLifetimeSeparated, clf, primaryTarget, trace.ZeroExpirationOps, 0)
	if err != nil {
		return SuiteReport{}, err
	}
	ctrlDogi = annotateRelative(ctrlBase, ctrlDogi)

	// 5. Optional labeled simulated S3 RTT comparison (alongside unthrottled runs).
	var simChecks []ModeMetrics
	if cfg.SimulatedRTT > 0 {
		sb, err := runOne("simulated-rtt", "baseline-fifo", ModeBaselineSingleStream, nil, primaryTarget, trace.HeldOutOps, cfg.SimulatedRTT)
		if err != nil {
			return SuiteReport{}, err
		}
		sd, err := runOne("simulated-rtt", "dogi-learned", ModeLifetimeSeparated, clf, primaryTarget, trace.HeldOutOps, cfg.SimulatedRTT)
		if err != nil {
			return SuiteReport{}, err
		}
		simChecks = append(simChecks, sb, annotateRelative(sb, sd))
	}

	return SuiteReport{
		Scale:              scaleName,
		TrainSamples:       len(trace.TrainRecords),
		HeldOutSamples:     len(trace.HeldOutRecords),
		HeldOutEval:        eval,
		PrimaryRuns:        []ModeMetrics{baseRes, learnedRes, oracleRes},
		AdversarialSweep:   advSweep,
		PackTargetSweep:    ptSweep,
		BoundaryControls:   []ModeMetrics{ctrlBase, ctrlDogi},
		SimulatedRTTChecks: simChecks,
	}, nil
}

// FormatCSV formats rep into a clean tabular CSV suitable for docs/dogi.csv.
func FormatCSV(rep SuiteReport) string {
	var b strings.Builder
	w := csv.NewWriter(&b)
	_ = w.Write([]string{
		"scenario", "name", "mode", "num_streams", "pack_target_bytes",
		"classifier_accuracy", "severe_mispred", "logical_chunks",
		"logical_plaintext_bytes", "logical_sealed_bytes",
		"physical_bytes_written", "physical_bytes_reclaimed", "active_physical_bytes",
		"live_bytes_copied", "write_amplification",
		"packs_created", "initial_packs_created", "compaction_packs_created", "packs_deleted", "pack_count_overhead_pct",
		"live_bytes_reduction_pct", "waf_reduction_pct",
		"put_requests", "get_range_requests", "delete_requests", "total_backend_requests",
		"wall_ms", "simulated_rtt_ms", "heap_alloc_delta_bytes",
	})
	appendRows := func(rows []ModeMetrics) {
		for _, r := range rows {
			_ = w.Write([]string{
				r.Scenario, r.Name, r.Mode,
				fmt.Sprintf("%d", r.NumStreams),
				fmt.Sprintf("%d", r.PackTargetBytes),
				fmt.Sprintf("%.4f", r.ClassifierAccuracy),
				fmt.Sprintf("%d", r.SevereMispredCount),
				fmt.Sprintf("%d", r.LogicalChunks),
				fmt.Sprintf("%d", r.LogicalPlaintextBytes),
				fmt.Sprintf("%d", r.LogicalSealedBytes),
				fmt.Sprintf("%d", r.PhysicalBytesWritten),
				fmt.Sprintf("%d", r.PhysicalBytesReclaimed),
				fmt.Sprintf("%d", r.ActivePhysicalBytes),
				fmt.Sprintf("%d", r.LiveBytesCopied),
				fmt.Sprintf("%.4f", r.WriteAmplification),
				fmt.Sprintf("%d", r.PacksCreated),
				fmt.Sprintf("%d", r.InitialPacksCreated),
				fmt.Sprintf("%d", r.CompactionPacksCreated),
				fmt.Sprintf("%d", r.PacksDeleted),
				fmt.Sprintf("%.2f", r.PackCountOverheadPct),
				fmt.Sprintf("%.2f", r.LiveBytesReductionPct),
				fmt.Sprintf("%.2f", r.WAFReductionPct),
				fmt.Sprintf("%d", r.PutRequests),
				fmt.Sprintf("%d", r.GetRangeRequests),
				fmt.Sprintf("%d", r.DeleteRequests),
				fmt.Sprintf("%d", r.TotalBackendRequests),
				fmt.Sprintf("%.3f", r.WallMs),
				fmt.Sprintf("%.3f", r.SimulatedRTTMs),
				fmt.Sprintf("%d", r.HeapAllocDeltaBytes),
			})
		}
	}
	appendRows(rep.PrimaryRuns)
	appendRows(rep.AdversarialSweep)
	appendRows(rep.PackTargetSweep)
	appendRows(rep.BoundaryControls)
	appendRows(rep.SimulatedRTTChecks)
	w.Flush()
	return b.String()
}

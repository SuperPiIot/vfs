// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package vfs_test

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/hanzoai/vfs"
	"github.com/hanzoai/vfs/internal/storetest"
	"github.com/hanzoai/vfs/pkg/backend"
	"github.com/hanzoai/vfs/pkg/backend/s3"
	"github.com/hanzoai/vfs/pkg/dogi"
	"github.com/hanzoai/vfs/pkg/fence"
)

func newS3Backend(t *testing.T, srv *storetest.S3, prefix string) backend.Backend {
	t.Helper()
	be, err := s3.New(s3.Config{
		Endpoint:  srv.Endpoint(),
		Insecure:  true,
		Region:    "us-east-1",
		Bucket:    srv.Bucket,
		Prefix:    prefix,
		AccessKey: "k",
		SecretKey: "s",
	})
	if err != nil {
		t.Fatal(err)
	}
	return storetest.Strict{Backend: be}
}

func TestDOGIHeldOutPredictorAndRealVFSIntegration(t *testing.T) {
	ciph := vfs.Hybrid(newKey(t))
	srv := storetest.NewS3(t, "dogi")

	// 1. Verify live vfs.Volume interoperability over the same S3 server and Hybrid cipher.
	vfsBE := newS3Backend(t, srv, "vfs-volume")
	if err := vfs.Create(ctx, cfg(vfsBE, ciph, 0)); err != nil {
		t.Fatal(err)
	}
	vol, err := vfs.Attach(ctx, cfg(vfsBE, ciph, 1))
	if err != nil {
		t.Fatal(err)
	}
	wsDir := mkdir(t, vol, vfs.RootInode, "workspace")
	srcFile := write(t, vol, wsDir, "main.go", []byte("package main\nfunc main() {}\n"))
	flush(t, vol)
	if got := string(readAll(t, reopen(t, vfsBE, ciph), srcFile.ID)); got != "package main\nfunc main() {}\n" {
		t.Fatalf("vfs.Volume round-trip mismatch: %q", got)
	}

	// 2. Generate coding-agent build + workspace trace and split strictly 40% train / 60% held-out.
	rounds := sized(8, 10)
	filesPerRound := sized(18, 28)
	avgChunk := sized(8<<10, 16<<10)
	packTarget := sized(32<<10, 64<<10)

	trace := dogi.GenerateAgentTrace(0x63D061, rounds, filesPerRound, avgChunk)
	clf := dogi.NewClassifier()
	clf.Train(trace.TrainRecords)

	eval := dogi.EvaluateClassifier(clf, trace.HeldOutRecords)
	if eval.Accuracy < 0.85 {
		t.Fatalf("held-out classifier accuracy = %.3f (< 0.85), confusion=%v", eval.Accuracy, eval.Confusion)
	}

	oracle := &dogi.OraclePredictor{Truth: trace.OracleTruth}

	type runSummary struct {
		name string
		st   dogi.Stats
		reqs int
	}

	executeTrace := func(prefix, name string, mode dogi.Mode, pred dogi.Predictor) runSummary {
		be := newS3Backend(t, srv, prefix)
		pk, err := dogi.NewPacker(ctx, dogi.Config{
			Backend:    be,
			Cipher:     ciph,
			Round:      1,
			Mode:       mode,
			PackTarget: packTarget,
			Predictor:  pred,
		})
		if err != nil {
			t.Fatalf("%s NewPacker: %v", name, err)
		}
		reqBefore := srv.Requests()
		expected := make(map[string][]byte)

		for _, op := range trace.HeldOutOps {
			switch op.Kind {
			case "put":
				if err := pk.Put(ctx, op.Key, op.Payload, op.Meta); err != nil {
					t.Fatalf("%s Put(%s): %v", name, op.Key, err)
				}
				expected[op.Key] = op.Payload
			case "delete":
				if err := pk.Delete(op.Key); err != nil {
					t.Fatalf("%s Delete(%s): %v", name, op.Key, err)
				}
				delete(expected, op.Key)
			case "checkpoint":
				if err := pk.Checkpoint(ctx); err != nil {
					t.Fatalf("%s Checkpoint: %v", name, err)
				}
			case "compact":
				if _, err := pk.Compact(ctx, dogi.CompactionPolicy{GarbageRatio: 0.40}); err != nil {
					t.Fatalf("%s Compact: %v", name, err)
				}
			}
		}
		if err := pk.Checkpoint(ctx); err != nil {
			t.Fatalf("%s final Checkpoint: %v", name, err)
		}
		if err := pk.CheckAccounting(); err != nil {
			t.Fatalf("%s CheckAccounting: %v", name, err)
		}

		// Reopen read-only from S3 "head" and verify surviving live objects byte-for-byte.
		ro, err := dogi.OpenPacker(ctx, dogi.Config{Backend: be, Cipher: ciph})
		if err != nil {
			t.Fatalf("%s OpenPacker: %v", name, err)
		}
		if err := ro.CheckAccounting(); err != nil {
			t.Fatalf("%s ro CheckAccounting: %v", name, err)
		}
		roKeys := ro.LiveKeys()
		if len(roKeys) != len(expected) {
			t.Fatalf("%s live keys = %d, want %d", name, len(roKeys), len(expected))
		}
		for i, k := range roKeys {
			if i%4 != 0 && i != len(roKeys)-1 {
				continue
			}
			got, err := ro.Get(ctx, k)
			if err != nil {
				t.Fatalf("%s ro.Get(%s): %v", name, k, err)
			}
			if !bytes.Equal(got, expected[k]) {
				t.Fatalf("%s payload mismatch for %s", name, k)
			}
		}

		return runSummary{
			name: name,
			st:   pk.Stats(),
			reqs: srv.Requests() - reqBefore,
		}
	}

	base := executeTrace("dogi-baseline", "baseline-fifo-1stream", dogi.ModeBaselineSingleStream, nil)
	learned := executeTrace("dogi-learned", "dogi-learned-3stream", dogi.ModeLifetimeSeparated, clf)
	oracleRun := executeTrace("dogi-oracle", "dogi-oracle-3stream", dogi.ModeLifetimeSeparated, oracle)

	if learned.st.LiveBytesCopied >= base.st.LiveBytesCopied {
		t.Fatalf("expected DOGI learned LiveBytesCopied (%d) < Baseline (%d)",
			learned.st.LiveBytesCopied, base.st.LiveBytesCopied)
	}
	if learned.st.WriteAmplification() >= base.st.WriteAmplification() {
		t.Fatalf("expected DOGI learned WAF (%.4f) < Baseline (%.4f)",
			learned.st.WriteAmplification(), base.st.WriteAmplification())
	}
	if oracleRun.st.LiveBytesCopied > learned.st.LiveBytesCopied {
		t.Fatalf("expected Oracle LiveBytesCopied (%d) <= Learned (%d)",
			oracleRun.st.LiveBytesCopied, learned.st.LiveBytesCopied)
	}

	packOverheadPct := (float64(learned.st.PacksCreated-base.st.PacksCreated) / float64(base.st.PacksCreated)) * 100.0
	t.Logf("Held-out accuracy=%.2f%% (train=%d heldout=%d severe_swaps=%d)",
		eval.Accuracy*100, len(trace.TrainRecords), len(trace.HeldOutRecords), eval.SevereMispredictions)
	t.Logf("Baseline FIFO : copied=%dB waf=%.4f packs=%d deleted=%d s3_reqs=%d",
		base.st.LiveBytesCopied, base.st.WriteAmplification(), base.st.PacksCreated, base.st.PacksDeleted, base.reqs)
	t.Logf("DOGI Learned  : copied=%dB waf=%.4f packs=%d (overhead=%+.1f%%) deleted=%d s3_reqs=%d",
		learned.st.LiveBytesCopied, learned.st.WriteAmplification(), learned.st.PacksCreated, packOverheadPct, learned.st.PacksDeleted, learned.reqs)
	t.Logf("DOGI Oracle   : copied=%dB waf=%.4f packs=%d deleted=%d s3_reqs=%d",
		oracleRun.st.LiveBytesCopied, oracleRun.st.WriteAmplification(), oracleRun.st.PacksCreated, oracleRun.st.PacksDeleted, oracleRun.reqs)
}

func TestDOGILostACKHeadCASAndRollback(t *testing.T) {
	ciph := vfs.Hybrid(newKey(t))
	srv := storetest.NewS3(t, "dogi-faults")
	rawBE := newS3Backend(t, srv, "vol-faults")

	var (
		beforeFn func(op, key string) error
		afterFn  func(op, key string) error
		hookMu   sync.Mutex
	)
	setHooks := func(b, a func(op, key string) error) {
		hookMu.Lock()
		beforeFn = b
		afterFn = a
		hookMu.Unlock()
	}
	hookBE := &storetest.Hook{
		Backend: rawBE,
		Before: func(op, key string) error {
			hookMu.Lock()
			fn := beforeFn
			hookMu.Unlock()
			if fn != nil {
				return fn(op, key)
			}
			return nil
		},
		After: func(op, key string) error {
			hookMu.Lock()
			fn := afterFn
			hookMu.Unlock()
			if fn != nil {
				return fn(op, key)
			}
			return nil
		},
	}

	pk, err := dogi.NewPacker(ctx, dogi.Config{
		Backend:    hookBE,
		Cipher:     ciph,
		Round:      1,
		Mode:       dogi.ModeLifetimeSeparated,
		PackTarget: 24 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Checkpoint 1 succeeds cleanly.
	keepA := content(101, 10<<10)
	keepB := content(102, 12<<10)
	if err := pk.Put(ctx, "/src/keep_a.go", keepA, dogi.ObjectMeta{Path: "/src/keep_a.go"}); err != nil {
		t.Fatal(err)
	}
	if err := pk.Put(ctx, "/src/keep_b.go", keepB, dogi.ObjectMeta{Path: "/src/keep_b.go"}); err != nil {
		t.Fatal(err)
	}
	if err := pk.Checkpoint(ctx); err != nil {
		t.Fatal(err)
	}
	if pk.Seq() != 1 {
		t.Fatalf("Seq = %d, want 1", pk.Seq())
	}

	// Phase A1: Inject Before failure on "head" -> atomic rollback, prior checkpoint intact.
	pendingC := content(103, 9<<10)
	if err := pk.Put(ctx, "/src/pending_c.go", pendingC, dogi.ObjectMeta{Path: "/src/pending_c.go"}); err != nil {
		t.Fatal(err)
	}
	errInjectedHead := errors.New("injected before head write")
	setHooks(func(op, key string) error {
		if key == "head" && (op == "putver" || op == "putabsent") {
			return errInjectedHead
		}
		return nil
	}, nil)
	if err := pk.Checkpoint(ctx); !errors.Is(err, errInjectedHead) {
		t.Fatalf("Checkpoint with Before(head) = %v, want %v", err, errInjectedHead)
	}
	setHooks(nil, nil)

	if pk.Seq() != 1 {
		t.Fatalf("Seq after rolled-back Checkpoint = %d, want 1", pk.Seq())
	}
	if err := pk.CheckAccounting(); err != nil {
		t.Fatalf("CheckAccounting after rollback: %v", err)
	}
	if pk.Stats().Rollbacks < 1 {
		t.Fatalf("expected Rollbacks >= 1, got %d", pk.Stats().Rollbacks)
	}

	// Fresh read-only reader must see Checkpoint 1 intact (keep_a, keep_b) and NOT pending_c.
	ro1, err := dogi.OpenPacker(ctx, dogi.Config{Backend: rawBE, Cipher: ciph})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ro1.Get(ctx, "/src/pending_c.go"); err == nil {
		t.Fatal("rolled-back pending_c leaked into committed head")
	}
	gotA, err := ro1.Get(ctx, "/src/keep_a.go")
	if err != nil || !bytes.Equal(gotA, keepA) {
		t.Fatalf("prior checkpoint keep_a damaged: err=%v", err)
	}

	// Retrying Checkpoint succeeds and commits pending_c cleanly.
	if err := pk.Checkpoint(ctx); err != nil {
		t.Fatalf("Checkpoint retry after rollback failed: %v", err)
	}
	if pk.Seq() != 2 {
		t.Fatalf("Seq after retry = %d, want 2", pk.Seq())
	}

	// Phase A2: Inject Before failure on pack/ during auto-flush in Put -> atomic rollback of Put.
	errInjectedPack := errors.New("injected before pack upload")
	setHooks(func(op, key string) error {
		if op == "putabsent" && strings.HasPrefix(key, "pack/") {
			return errInjectedPack
		}
		return nil
	}, nil)
	bigD := content(104, 28<<10) // exceeds 24 KiB PackTarget, triggers auto-flush
	if err := pk.Put(ctx, "/target/fail_d.o", bigD, dogi.ObjectMeta{Path: "/target/fail_d.o"}); !errors.Is(err, errInjectedPack) {
		t.Fatalf("Put auto-flush with Before(pack) = %v, want %v", err, errInjectedPack)
	}
	setHooks(nil, nil)
	if err := pk.CheckAccounting(); err != nil {
		t.Fatalf("CheckAccounting after Put rollback: %v", err)
	}
	if _, err := pk.Get(ctx, "/target/fail_d.o"); err == nil {
		t.Fatal("failed Put object fail_d still present in live map")
	}

	// Phase B: Inject After lost-ACK on pack/ PutIfAbsent -> clean recovery preserving all objects.
	var packLostOnce atomic.Bool
	packLostOnce.Store(true)
	setHooks(nil, func(op, key string) error {
		if op == "putabsent" && strings.HasPrefix(key, "pack/") && packLostOnce.Swap(false) {
			return errors.New("simulated lost ACK after pack PutIfAbsent")
		}
		return nil
	})
	recovE := content(105, 28<<10)
	if err := pk.Put(ctx, "/target/recov_e.o", recovE, dogi.ObjectMeta{Path: "/target/recov_e.o"}); err != nil {
		t.Fatalf("Put with After(pack) lost-ACK failed: %v", err)
	}
	if err := pk.Checkpoint(ctx); err != nil {
		t.Fatalf("Checkpoint after pack lost-ACK failed: %v", err)
	}
	setHooks(nil, nil)
	if pk.Stats().LostACKRecoveries < 1 {
		t.Fatalf("expected LostACKRecoveries >= 1 after pack lost-ACK, got %d", pk.Stats().LostACKRecoveries)
	}

	// Phase C: Inject After lost-ACK on "head" PutIfVersion -> landed fenced token recognized cleanly.
	var headLostOnce atomic.Bool
	headLostOnce.Store(true)
	setHooks(nil, func(op, key string) error {
		if key == "head" && (op == "putver" || op == "putabsent") && headLostOnce.Swap(false) {
			return errors.New("simulated lost ACK after head CAS")
		}
		return nil
	})
	recovF := content(106, 8<<10)
	if err := pk.Put(ctx, "/src/recov_f.go", recovF, dogi.ObjectMeta{Path: "/src/recov_f.go"}); err != nil {
		t.Fatal(err)
	}
	beforeRecov := pk.Stats().LostACKRecoveries
	if err := pk.Checkpoint(ctx); err != nil {
		t.Fatalf("Checkpoint with After(head) lost-ACK failed: %v", err)
	}
	setHooks(nil, nil)
	if pk.Stats().LostACKRecoveries <= beforeRecov {
		t.Fatalf("expected LostACKRecoveries to increase on head lost-ACK, before=%d after=%d",
			beforeRecov, pk.Stats().LostACKRecoveries)
	}
	ro3, err := dogi.OpenPacker(ctx, dogi.Config{Backend: rawBE, Cipher: ciph})
	if err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string][]byte{
		"/src/keep_a.go":    keepA,
		"/src/keep_b.go":    keepB,
		"/src/pending_c.go": pendingC,
		"/target/recov_e.o": recovE,
		"/src/recov_f.go":   recovF,
	} {
		got, err := ro3.Get(ctx, k)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("ro3.Get(%s): err=%v match=%v", k, err, bytes.Equal(got, want))
		}
	}

	// Phase D: 412 CAS fence.ErrStaleRound when a higher Round takes ownership on storetest.NewS3.
	if _, err := dogi.AttachPacker(ctx, dogi.Config{
		Backend:    rawBE,
		Cipher:     ciph,
		Round:      1, // equal to recorded round 1 -> must fail
		Mode:       dogi.ModeLifetimeSeparated,
		PackTarget: 24 << 10,
	}); !errors.Is(err, fence.ErrStaleRound) {
		t.Fatalf("AttachPacker at equal round 1 = %v, want fence.ErrStaleRound", err)
	}

	pk2, err := dogi.AttachPacker(ctx, dogi.Config{
		Backend:    rawBE,
		Cipher:     ciph,
		Round:      2,
		Mode:       dogi.ModeLifetimeSeparated,
		PackTarget: 24 << 10,
	})
	if err != nil {
		t.Fatalf("AttachPacker at round 2: %v", err)
	}

	// Deposed writer pk (Round 1) must be fenced with fence.ErrStaleRound via S3 412 CAS.
	if err := pk.Put(ctx, "/src/stale_write.go", content(107, 4<<10), dogi.ObjectMeta{Path: "/src/stale_write.go"}); err != nil {
		t.Fatal(err)
	}
	if err := pk.Checkpoint(ctx); !errors.Is(err, fence.ErrStaleRound) {
		t.Fatalf("deposed writer Checkpoint = %v, want fence.ErrStaleRound", err)
	}
	if err := pk.CheckAccounting(); err != nil {
		t.Fatalf("CheckAccounting after stale round fence: %v", err)
	}

	// New owner pk2 still has all committed objects and can advance head cleanly.
	gotF, err := pk2.Get(ctx, "/src/recov_f.go")
	if err != nil || !bytes.Equal(gotF, recovF) {
		t.Fatalf("new owner lost recov_f: %v", err)
	}
	if _, err := pk2.Get(ctx, "/src/stale_write.go"); err == nil {
		t.Fatal("deposed writer object leaked into new owner")
	}
	if err := pk2.Put(ctx, "/src/owner2.go", content(108, 4<<10), dogi.ObjectMeta{Path: "/src/owner2.go"}); err != nil {
		t.Fatal(err)
	}
	if err := pk2.Checkpoint(ctx); err != nil {
		t.Fatalf("owner2 Checkpoint: %v", err)
	}
}

func TestDOGIConcurrentReadDuringGCRelocation(t *testing.T) {
	ciph := vfs.Hybrid(newKey(t))
	srv := storetest.NewS3(t, "dogi-conc")
	be := newS3Backend(t, srv, "conc-vol")

	// Use ModeBaselineSingleStream so live chunks and ephemeral chunks are co-packed,
	// forcing Compact to relocate every live chunk and delete old packs while
	// concurrent Get goroutines actively read live keys.
	pk, err := dogi.NewPacker(ctx, dogi.Config{
		Backend:    be,
		Cipher:     ciph,
		Round:      1,
		Mode:       dogi.ModeBaselineSingleStream,
		PackTarget: 24 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	const numLive = 16
	liveKeys := make([]string, numLive)
	liveData := make([][]byte, numLive)
	for i := range numLive {
		liveKeys[i] = fmt.Sprintf("/src/live_%02d.go", i)
		liveData[i] = content(uint64(2000+i), 5<<10)
		if err := pk.Put(ctx, liveKeys[i], liveData[i], dogi.ObjectMeta{Path: liveKeys[i]}); err != nil {
			t.Fatal(err)
		}
		// Interleave ephemeral chunks in the same packs so each pack becomes >65% dead once deleted.
		for j := range 3 {
			ek := fmt.Sprintf("/target/ephem_init_%02d_%d.o", i, j)
			if err := pk.Put(ctx, ek, content(uint64(3000+i*10+j), 4<<10), dogi.ObjectMeta{Path: ek}); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := pk.Checkpoint(ctx); err != nil {
		t.Fatal(err)
	}
	for i := range numLive {
		for j := range 3 {
			if err := pk.Delete(fmt.Sprintf("/target/ephem_init_%02d_%d.o", i, j)); err != nil {
				t.Fatal(err)
			}
		}
	}

	// Launch concurrent Get readers hammering liveKeys throughout compaction & deletion cycles.
	stopReaders := make(chan struct{})
	var readOps atomic.Int64
	var readErrs atomic.Int64
	var readerWG sync.WaitGroup

	readers := sized(4, 8)
	for r := range readers {
		readerWG.Add(1)
		go func(id int) {
			defer readerWG.Done()
			for step := 0; ; step++ {
				select {
				case <-stopReaders:
					return
				default:
				}
				idx := (id + step) % numLive
				got, err := pk.Get(ctx, liveKeys[idx])
				if err != nil {
					readErrs.Add(1)
					t.Errorf("reader %d Get(%s): %v", id, liveKeys[idx], err)
					return
				}
				if !bytes.Equal(got, liveData[idx]) {
					readErrs.Add(1)
					t.Errorf("reader %d data mismatch on %s", id, liveKeys[idx])
					return
				}
				readOps.Add(1)
			}
		}(r)
	}

	// Run multiple compaction + fresh churn cycles while readers are active.
	var totalRelocated int
	var totalDeleted int
	cycles := sized(3, 5)
	for c := range cycles {
		res, err := pk.Compact(ctx, dogi.CompactionPolicy{GarbageRatio: 0.40})
		if err != nil {
			close(stopReaders)
			readerWG.Wait()
			t.Fatalf("Compact cycle %d: %v", c, err)
		}
		totalRelocated += res.ChunksRelocated
		totalDeleted += res.PacksDeleted

		// Inject a fresh batch of dead-to-be ephemeral + rewritten live keys for next cycle.
		for j := range 12 {
			ek := fmt.Sprintf("/target/ephem_c%d_%02d.o", c, j)
			if err := pk.Put(ctx, ek, content(uint64(4000+c*100+j), 6<<10), dogi.ObjectMeta{Path: ek}); err != nil {
				close(stopReaders)
				readerWG.Wait()
				t.Fatal(err)
			}
		}
		if err := pk.Checkpoint(ctx); err != nil {
			close(stopReaders)
			readerWG.Wait()
			t.Fatal(err)
		}
		for j := range 12 {
			if err := pk.Delete(fmt.Sprintf("/target/ephem_c%d_%02d.o", c, j)); err != nil {
				close(stopReaders)
				readerWG.Wait()
				t.Fatal(err)
			}
		}
	}
	res, err := pk.Compact(ctx, dogi.CompactionPolicy{GarbageRatio: 0.40})
	if err != nil {
		close(stopReaders)
		readerWG.Wait()
		t.Fatalf("final Compact: %v", err)
	}
	totalRelocated += res.ChunksRelocated
	totalDeleted += res.PacksDeleted

	close(stopReaders)
	readerWG.Wait()

	if readErrs.Load() > 0 {
		t.Fatalf("concurrent readers observed %d errors", readErrs.Load())
	}
	if readOps.Load() < 50 {
		t.Fatalf("too few concurrent reads (%d) observed during GC relocation", readOps.Load())
	}
	if totalRelocated == 0 || totalDeleted == 0 {
		t.Fatalf("expected active GC relocation & deletion: relocated=%d deleted=%d", totalRelocated, totalDeleted)
	}
	if err := pk.CheckAccounting(); err != nil {
		t.Fatalf("CheckAccounting after concurrent GC: %v", err)
	}
	t.Logf("Concurrent GC reads passed: read_ops=%d relocated_chunks=%d packs_deleted=%d waf=%.4f",
		readOps.Load(), totalRelocated, totalDeleted, pk.Stats().WriteAmplification())
}

// Copyright 2026 Hanzo AI Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

package dogi_test

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/luxfi/age"

	"github.com/hanzoai/vfs"
	"github.com/hanzoai/vfs/internal/blake3"
	"github.com/hanzoai/vfs/internal/storetest"
	"github.com/hanzoai/vfs/pkg/backend"
	"github.com/hanzoai/vfs/pkg/dogi"
)

func testCipher(t *testing.T) dogi.Cipher {
	t.Helper()
	k, err := age.GenerateHybridIdentity()
	if err != nil {
		t.Fatal(err)
	}
	return vfs.Hybrid(k)
}

func TestClassifierHeldOutAndNoOracleLeakage(t *testing.T) {
	trace := dogi.GenerateAgentTrace(0x1344, 10, 25, 16<<10)
	if len(trace.TrainRecords) == 0 || len(trace.HeldOutRecords) == 0 {
		t.Fatalf("empty splits: train=%d heldout=%d", len(trace.TrainRecords), len(trace.HeldOutRecords))
	}

	clf := dogi.NewClassifier()
	if clf.IsTrained() {
		t.Fatal("fresh classifier should not be trained")
	}
	// Untrained neutral prediction.
	if got := clf.Classify(trace.HeldOutRecords[0].Meta); got != dogi.LifetimeMedium {
		t.Fatalf("untrained Classify = %v, want medium", got)
	}

	clf.Train(trace.TrainRecords)
	if !clf.IsTrained() || clf.SamplesSeen() != len(trace.TrainRecords) {
		t.Fatalf("trained=%v samples=%d want %d", clf.IsTrained(), clf.SamplesSeen(), len(trace.TrainRecords))
	}

	eval := dogi.EvaluateClassifier(clf, trace.HeldOutRecords)
	if eval.Accuracy < 0.85 {
		t.Fatalf("held-out accuracy = %.3f (< 0.85), confusion=%v", eval.Accuracy, eval.Confusion)
	}

	// Verify ObjectMeta carries no ground truth and ClassifyOracle is strictly explicit.
	for i := range 10 {
		rec := trace.HeldOutRecords[i]
		if got := dogi.ClassifyOracle(rec); got != rec.TrueLifetime {
			t.Fatalf("ClassifyOracle = %v, want %v", got, rec.TrueLifetime)
		}
		probs := clf.PredictProba(rec.Meta)
		sum := probs[0] + probs[1] + probs[2]
		if sum < 0.999 || sum > 1.001 {
			t.Fatalf("PredictProba sum = %v", sum)
		}
	}

	// Online update smoke test.
	clf.UpdateOnline(trace.HeldOutRecords[0])
	if clf.SamplesSeen() != len(trace.TrainRecords)+1 {
		t.Fatalf("SamplesSeen after online update = %d", clf.SamplesSeen())
	}
}

func TestClassifierAdversarialNoiseEvaluation(t *testing.T) {
	trace := dogi.GenerateAgentTrace(0xBEEF, 8, 20, 8<<10)
	clf := dogi.NewClassifier()
	clf.Train(trace.TrainRecords)

	clean := dogi.EvaluateClassifier(clf, trace.HeldOutRecords)
	noisy := dogi.EvaluateClassifier(dogi.NewAdversarialPredictor(clf, 0.35, 99), trace.HeldOutRecords)
	if noisy.Accuracy >= clean.Accuracy {
		t.Fatalf("adversarial 35%% noise did not degrade accuracy: clean=%.3f noisy=%.3f", clean.Accuracy, noisy.Accuracy)
	}
}

func TestPackerCiphertextOnlyAndDedupConflictRefusal(t *testing.T) {
	ctx := context.Background()
	mem := storetest.NewMem()
	strict := storetest.Strict{Backend: mem}
	c := testCipher(t)

	pk, err := dogi.NewPacker(ctx, dogi.Config{
		Backend:    strict,
		Cipher:     c,
		Round:      1,
		Mode:       dogi.ModeLifetimeSeparated,
		PackTarget: 32 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	secretName := "/ws-secret/models/confidential_weights_9981.safetensors"
	secretBody := bytes.Repeat([]byte("SECRET-PLAINTEXT-PAYLOAD-1234567890"), 300)
	if err := pk.Put(ctx, secretName, secretBody, dogi.ObjectMeta{Path: secretName, Size: int64(len(secretBody))}); err != nil {
		t.Fatal(err)
	}
	if err := pk.Checkpoint(ctx); err != nil {
		t.Fatal(err)
	}
	if err := pk.CheckAccounting(); err != nil {
		t.Fatal(err)
	}

	// Rule 1: Zero plaintext key or body leakage into backend storage.
	for k, raw := range mem.Dump() {
		if err := backend.ValidKey(k); err != nil {
			t.Fatalf("stored key %q invalid: %v", k, err)
		}
		if strings.Contains(k, "confidential") || strings.Contains(k, "safetensors") {
			t.Fatalf("backend key leaked plaintext path: %q", k)
		}
		if bytes.Contains(raw, []byte("SECRET-PLAINTEXT-PAYLOAD")) || bytes.Contains(raw, []byte("confidential_weights")) {
			t.Fatalf("backend object %q leaked plaintext content/path", k)
		}
	}

	// Read back via read-only OpenPacker.
	ro, err := dogi.OpenPacker(ctx, dogi.Config{Backend: strict, Cipher: c})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ro.Get(ctx, secretName)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, secretBody) {
		t.Fatal("read back payload mismatch")
	}

	// Verify immutable pack/<2hex>/<blake3> conflict refusal if existing payload size differs.
	// Use a deterministic authenticated cipher to force an exact key collision check.
	detC := &deterministicTestCipher{key: [32]byte{1, 2, 3, 4, 5, 6, 7, 8}}
	pkDet, err := dogi.NewPacker(ctx, dogi.Config{
		Backend:    strict,
		Cipher:     detC,
		Round:      1,
		Mode:       dogi.ModeBaselineSingleStream,
		PackTarget: 8 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	payload := bytes.Repeat([]byte("A"), 9000)
	// Pre-plant a differing-size payload at the deterministic pack key pkDet would compute.
	pkDetTmp, err := dogi.NewPacker(ctx, dogi.Config{
		Backend:    storetest.NewMem(),
		Cipher:     detC,
		Round:      1,
		VolumeID:   pkDet.VolumeID(),
		Mode:       dogi.ModeBaselineSingleStream,
		PackTarget: 8 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := pkDetTmp.Put(ctx, "/k", payload, dogi.ObjectMeta{Path: "/k", Size: int64(len(payload))}); err != nil {
		t.Fatal(err)
	}
	targetPackKey := pkDetTmp.Packs()[0].Key
	// Pre-populate conflicting size in mem at targetPackKey.
	if err := mem.Put(ctx, targetPackKey, []byte("conflicting-short-payload")); err != nil {
		t.Fatal(err)
	}
	err = pkDet.Put(ctx, "/k", payload, dogi.ObjectMeta{Path: "/k", Size: int64(len(payload))})
	if !errors.Is(err, dogi.ErrPackConflict) {
		t.Fatalf("expected ErrPackConflict on differing-size pack key, got %v", err)
	}
	// Verify existing object was NOT overwritten.
	after, _, _ := mem.Get(ctx, targetPackKey, backend.Unbounded)
	if string(after) != "conflicting-short-payload" {
		t.Fatal("ErrPackConflict overwrote existing object")
	}
}

type deterministicTestCipher struct {
	key [32]byte
}

func (d *deterministicTestCipher) Seal(p []byte) ([]byte, string, error) {
	pad := make([]byte, len(p))
	h := blake3.Keyed(&d.key)
	_, _ = h.Write(p)
	tag := h.Sum(nil)
	// Deterministic stream keyed by tag so ciphertext depends deterministically on plaintext.
	k := blake3.Keyed(&d.key)
	_, _ = k.Write(tag)
	k.Expand(pad)
	out := make([]byte, len(p)+len(tag))
	for i := range p {
		out[i] = p[i] ^ pad[i]
	}
	copy(out[len(p):], tag)
	return out, "det1", nil
}

func (d *deterministicTestCipher) Open(s []byte, id string) ([]byte, error) {
	if id != "det1" || len(s) < blake3.Size {
		return nil, errors.New("bad key or truncated ciphertext")
	}
	n := len(s) - blake3.Size
	ct, wantTag := s[:n], s[n:]
	pad := make([]byte, n)
	k := blake3.Keyed(&d.key)
	_, _ = k.Write(wantTag)
	k.Expand(pad)
	plain := make([]byte, n)
	for i := range plain {
		plain[i] = ct[i] ^ pad[i]
	}
	h := blake3.Keyed(&d.key)
	_, _ = h.Write(plain)
	gotTag := h.Sum(nil)
	if !bytes.Equal(gotTag, wantTag) {
		return nil, errors.New("tampered ciphertext")
	}
	return plain, nil
}

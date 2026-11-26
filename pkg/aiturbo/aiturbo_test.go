package aiturbo

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hanzoai/vfs/pkg/backend"
	_ "github.com/hanzoai/vfs/pkg/backend/file"
)

func newTestBackend(t *testing.T) backend.Backend {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "store")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	be, err := backend.Open(context.Background(), "file://"+dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = be.Close() })
	return be
}

func TestPlannerBatchReductionRatio(t *testing.T) {
	p, err := NewPlanner(Options{GroupSize: 8})
	if err != nil {
		t.Fatalf("NewPlanner: %v", err)
	}

	const nItems = 256
	items := make([]Item, nItems)
	for i := 0; i < nItems; i++ {
		items[i] = Item{
			Key:  fmt.Sprintf("pack/%02x/%064x", i%16, i),
			Data: make([]byte, 4096),
		}
	}

	plan, err := p.PlanBatches(items)
	if err != nil {
		t.Fatalf("PlanBatches: %v", err)
	}

	// 256 chunks / 8 per batch = exactly 32 requests
	if plan.RequestsCount != 32 {
		t.Errorf("got %d requests, want 32", plan.RequestsCount)
	}
	if len(plan.Batches) != 32 {
		t.Errorf("got %d batches, want 32", len(plan.Batches))
	}
	reduction := float64(plan.TotalItems) / float64(plan.RequestsCount)
	if reduction != 8.0 {
		t.Errorf("got reduction ratio %f, want 8.0", reduction)
	}
}

func TestHeadConditionalPUTStrictSeparation(t *testing.T) {
	p, err := NewPlanner(Options{GroupSize: 8})
	if err != nil {
		t.Fatal(err)
	}

	items := []Item{
		{Key: "pack/00/abc", Data: []byte("chunk-data")},
		{Key: "head", Data: []byte("head-record")}, // VIOLATION
		{Key: "pack/01/def", Data: []byte("chunk-data")},
	}

	_, err = p.PlanBatches(items)
	if err != ErrHeadCoalesceForbidden {
		t.Fatalf("expected ErrHeadCoalesceForbidden, got %v", err)
	}
}

func TestGroupedUploadAndDownloadRoundtrip(t *testing.T) {
	be := newTestBackend(t)
	ctx := context.Background()

	opt := Options{
		GroupSize: 8,
		Streams:   4,
	}

	uploader, err := NewGroupedUploader(ctx, be, opt)
	if err != nil {
		t.Fatal(err)
	}

	const totalObjs = 64
	items := make([]Item, totalObjs)
	var doneCount atomic.Int32

	for i := 0; i < totalObjs; i++ {
		key := fmt.Sprintf("meta/%02x/obj-%04d", i%16, i)
		data := []byte(fmt.Sprintf("content-payload-%04d-with-padding", i))
		items[i] = Item{
			Key:  key,
			Data: data,
			Done: func() {
				doneCount.Add(1)
			},
		}
	}

	plan, err := uploader.planner.PlanBatches(items)
	if err != nil {
		t.Fatal(err)
	}

	metrics, err := uploader.UploadPlan(ctx, plan)
	if err != nil {
		t.Fatalf("UploadPlan: %v", err)
	}

	if doneCount.Load() != totalObjs {
		t.Errorf("done count = %d, want %d", doneCount.Load(), totalObjs)
	}
	if metrics.RequestsIssued != 8 {
		t.Errorf("requests issued = %d, want 8", metrics.RequestsIssued)
	}

	// Now verify with GroupedDownloader
	downloader, err := NewGroupedDownloader(ctx, be, opt)
	if err != nil {
		t.Fatal(err)
	}

	keys := make([]string, totalObjs)
	for i := 0; i < totalObjs; i++ {
		keys[i] = items[i].Key
	}

	fetched, dMetrics, err := downloader.DownloadKeys(ctx, keys)
	if err != nil {
		t.Fatalf("DownloadKeys: %v", err)
	}

	if len(fetched) != totalObjs {
		t.Errorf("fetched %d items, want %d", len(fetched), totalObjs)
	}
	if dMetrics.RequestsIssued != 8 {
		t.Errorf("download requests issued = %d, want 8", dMetrics.RequestsIssued)
	}

	for i := 0; i < totalObjs; i++ {
		got := fetched[items[i].Key]
		if !bytes.Equal(got, items[i].Data) {
			t.Errorf("item %d payload mismatch: got %q, want %q", i, got, items[i].Data)
		}
	}
}

func TestSimulatedNetworkThroughputAdvantage(t *testing.T) {
	be := newTestBackend(t)
	ctx := context.Background()

	const (
		numChunks = 16
		simRTT    = 5 * time.Millisecond
	)

	items := make([]Item, numChunks)
	for i := 0; i < numChunks; i++ {
		items[i] = Item{
			Key:  fmt.Sprintf("pack/%02x/bench-%02d", i%16, i),
			Data: make([]byte, 1024),
		}
	}

	// 1. Grouped execution (groupSize=8 -> 2 requests)
	gUploader, err := NewGroupedUploader(ctx, be, Options{
		GroupSize: 8,
		Streams:   1, // Serialized to isolate RTT effects
		SimRTT:    simRTT,
	})
	if err != nil {
		t.Fatal(err)
	}

	planGrouped, err := gUploader.planner.PlanBatches(items)
	if err != nil {
		t.Fatal(err)
	}

	startGrouped := time.Now()
	_, err = gUploader.UploadPlan(ctx, planGrouped)
	if err != nil {
		t.Fatal(err)
	}
	durGrouped := time.Since(startGrouped)

	// 2. Baseline execution (groupSize=1 -> 16 requests)
	bUploader, err := NewGroupedUploader(ctx, be, Options{
		GroupSize: 1,
		Streams:   1,
		SimRTT:    simRTT,
	})
	if err != nil {
		t.Fatal(err)
	}

	planBaseline, err := bUploader.planner.PlanBatches(items)
	if err != nil {
		t.Fatal(err)
	}

	startBaseline := time.Now()
	_, err = bUploader.UploadPlan(ctx, planBaseline)
	if err != nil {
		t.Fatal(err)
	}
	durBaseline := time.Since(startBaseline)

	// With SimRTT=5ms, 16 requests should take >= 80ms, while 2 requests take ~10ms
	t.Logf("Baseline (16 reqs): %v, Grouped (2 reqs): %v, Speedup: %.2fx",
		durBaseline, durGrouped, float64(durBaseline)/float64(durGrouped))

	if durGrouped >= durBaseline {
		t.Errorf("expected grouped (%v) to be faster than baseline (%v)", durGrouped, durBaseline)
	}
}

// Command aiturbobench benchmarks baseline 1-by-1 S3 checkpoint publication
// against AITURBO grouped IO planning (FAST '26).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/hanzoai/vfs/pkg/aiturbo"
	"github.com/hanzoai/vfs/pkg/backend"
	_ "github.com/hanzoai/vfs/pkg/backend/file"
)

func main() {
	chunks := flag.Int("chunks", 256, "number of 4 MiB chunks to simulate (256 = 1 GiB)")
	groupSize := flag.Int("group-size", 8, "AITURBO grouping factor")
	streams := flag.Int("streams", 16, "concurrent upload streams")
	rttMs := flag.Int("rtt", 5, "simulated round-trip latency in ms")
	csvOut := flag.String("csv", "", "optional path to append CSV results")
	flag.Parse()

	ctx := context.Background()
	dir, err := os.MkdirTemp("", "aiturbobench-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create temp dir: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(dir)

	be, err := backend.Open(ctx, "file://"+dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open backend: %v\n", err)
		os.Exit(1)
	}
	defer be.Close()

	// 4 MiB dummy payload per chunk
	chunkBytes := 4 * 1024 * 1024
	payload := make([]byte, chunkBytes)

	items := make([]aiturbo.Item, *chunks)
	for i := 0; i < *chunks; i++ {
		items[i] = aiturbo.Item{
			Key:  fmt.Sprintf("pack/%02x/chunk-%04d", i%16, i),
			Data: payload,
		}
	}

	totalGiB := float64(*chunks*chunkBytes) / (1024 * 1024 * 1024)
	rtt := time.Duration(*rttMs) * time.Millisecond

	fmt.Printf("=== AITURBO Checkpoint IO Benchmark (FAST '26) ===\n")
	fmt.Printf("Volume Data: %.2f GiB (%d chunks x 4 MiB)\n", totalGiB, *chunks)
	fmt.Printf("Simulated RTT: %v | Streams: %d | Group Factor: %dx\n\n", rtt, *streams, *groupSize)

	// 1. Run Baseline (GroupSize = 1)
	baseUploader, err := aiturbo.NewGroupedUploader(ctx, be, aiturbo.Options{
		GroupSize: 1,
		Streams:   *streams,
		SimRTT:    rtt,
	})
	if err != nil {
		panic(err)
	}

	planBase, _ := baseUploader.Planner().PlanBatches(items)
	baseMetrics, err := baseUploader.UploadPlan(ctx, planBase)
	if err != nil {
		panic(err)
	}

	// Clean store for fair grouped run
	_ = os.RemoveAll(filepath.Join(dir, "pack"))

	// 2. Run AITURBO Grouped
	turboUploader, err := aiturbo.NewGroupedUploader(ctx, be, aiturbo.Options{
		GroupSize: *groupSize,
		Streams:   *streams,
		SimRTT:    rtt,
	})
	if err != nil {
		panic(err)
	}

	planTurbo, _ := turboUploader.Planner().PlanBatches(items)
	turboMetrics, err := turboUploader.UploadPlan(ctx, planTurbo)
	if err != nil {
		panic(err)
	}

	speedup := baseMetrics.Duration.Seconds() / turboMetrics.Duration.Seconds()
	reduction := float64(baseMetrics.RequestsIssued) / float64(turboMetrics.RequestsIssued)

	fmt.Printf("%-24s | %-12s | %-14s | %-14s | %-10s\n", "Policy", "Requests/GiB", "Duration", "Throughput", "Speedup")
	fmt.Println("-------------------------------------------------------------------------------------")
	fmt.Printf("%-24s | %-12d | %-14v | %-11.2f MB/s | baseline\n",
		"baseline-s3 (1-by-1)",
		int(float64(baseMetrics.RequestsIssued)/totalGiB),
		baseMetrics.Duration.Round(time.Millisecond),
		baseMetrics.BytesPerSec/(1024*1024),
	)
	fmt.Printf("%-24s | %-12d | %-14v | %-11.2f MB/s | %.2fx\n",
		fmt.Sprintf("aiturbo-grouped (%dx)", *groupSize),
		int(float64(turboMetrics.RequestsIssued)/totalGiB),
		turboMetrics.Duration.Round(time.Millisecond),
		turboMetrics.BytesPerSec/(1024*1024),
		speedup,
	)

	fmt.Printf("\nS3 Request Reduction: %.1fx (%d -> %d requests)\n",
		reduction, baseMetrics.RequestsIssued, turboMetrics.RequestsIssued)

	if *csvOut != "" {
		f, err := os.OpenFile(*csvOut, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err == nil {
			defer f.Close()
			fmt.Fprintf(f, "%d,%d,%d,%d,%v,%v,%.2f,%.2f\n",
				*chunks, *groupSize, *streams, *rttMs,
				baseMetrics.Duration, turboMetrics.Duration,
				speedup, reduction,
			)
		}
	}
}

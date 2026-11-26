package aiturbo

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hanzoai/vfs/pkg/backend"
)

// GroupedDownloader provides aggregated, pipelined object retrieval for checkpoint restores.
type GroupedDownloader struct {
	ctx     context.Context
	be      backend.Backend
	opt     Options
	planner *Planner

	requestsIssued atomic.Int64
	bytesRead      atomic.Int64
}

// NewGroupedDownloader initializes a downloader wrapping a Backend.
func NewGroupedDownloader(ctx context.Context, be backend.Backend, opt Options) (*GroupedDownloader, error) {
	p, err := NewPlanner(opt)
	if err != nil {
		return nil, err
	}
	return &GroupedDownloader{
		ctx:     ctx,
		be:      be,
		opt:     p.opt,
		planner: p,
	}, nil
}

// DownloadKeys fetches all named keys in grouped batches into results.
func (d *GroupedDownloader) DownloadKeys(ctx context.Context, keys []string) (map[string][]byte, *Metrics, error) {
	if len(keys) == 0 {
		return nil, nil, ErrEmptyBatch
	}

	start := time.Now()

	// Partition keys into batches
	var batches [][]string
	for i := 0; i < len(keys); i += d.opt.GroupSize {
		end := i + d.opt.GroupSize
		if end > len(keys) {
			end = len(keys)
		}
		batches = append(batches, keys[i:end])
	}

	resMap := make(map[string][]byte, len(keys))
	var resMu sync.Mutex

	sem := make(chan struct{}, d.opt.Streams)
	var wg sync.WaitGroup
	var (
		errMu    sync.Mutex
		firstErr error
	)

	fail := func(err error) {
		errMu.Lock()
		defer errMu.Unlock()
		if firstErr == nil {
			firstErr = err
		}
	}

	for _, batch := range batches {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}

		errMu.Lock()
		if firstErr != nil {
			errMu.Unlock()
			<-sem
			break
		}
		errMu.Unlock()

		wg.Add(1)
		go func(keyBatch []string) {
			defer wg.Done()
			defer func() { <-sem }()

			if d.opt.SimRTT > 0 {
				select {
				case <-time.After(d.opt.SimRTT):
				case <-ctx.Done():
					fail(ctx.Err())
					return
				}
			}

			// 1 Grouped Request Descriptor
			d.requestsIssued.Add(1)

			localRes := make(map[string][]byte, len(keyBatch))
			for _, k := range keyBatch {
				data, _, err := d.be.Get(ctx, k, backend.Unbounded)
				if err != nil {
					fail(fmt.Errorf("aiturbo: get %q: %w", k, err))
					return
				}
				localRes[k] = data
				d.bytesRead.Add(int64(len(data)))
			}

			resMu.Lock()
			for k, v := range localRes {
				resMap[k] = v
			}
			resMu.Unlock()
		}(batch)
	}

	wg.Wait()

	if firstErr != nil {
		return nil, nil, firstErr
	}

	dur := time.Since(start)
	if dur == 0 {
		dur = time.Microsecond
	}

	reqs := int(d.requestsIssued.Load())
	bytes := d.bytesRead.Load()

	m := &Metrics{
		TotalBytes:       bytes,
		TotalObjects:     len(keys),
		RequestsIssued:   reqs,
		RequestReduction: float64(len(keys)) / float64(max(reqs, 1)),
		Duration:         dur,
		BytesPerSec:      float64(bytes) / dur.Seconds(),
	}

	return resMap, m, nil
}

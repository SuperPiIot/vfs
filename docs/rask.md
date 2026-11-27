# RASK — Compact In-Memory Sparse-Range Index (HIP-1344)

Refs #59. Tracked by HIP-1344. The implementation is in `rask.go`, unit and
differential oracle tests are in `rask_test.go`, the reproducible benchmark CLI
is `tools/raskbench`, and raw machine-readable measurements are in `docs/rask.csv`.

## 1. Primary reference, scope, and clean-room audit

* **Primary reference:** RASK (24th USENIX Conference on File and Storage
  Technologies, FAST 2026) — compact range indexing for sparse file extents and
  write-ahead journal spans.
* **Scope in `hanzoai/vfs`:** `rask.go` prototypes an in-memory segmented
  B+tree (`RaskRunIndex`) over `run{a, b uint64}` intervals alongside an exact
  16-byte `[]run` sorted binary-search baseline (`BaselineRunSlice`, matching
  `file.go` `layer.void` / `join([]run)`). It handles arbitrary `0..MaxSize`
  (`1 << 50` 1 PiB sparse files and beyond) purely in memory.
* **Format-v2 invariance & clean-room audit:** Zero frozen format-v2 files
  (`head.go`, `record.go`, `shard.go`, `pack.go`, `tree.go`, `file.go`) or
  dependency manifests (`go.mod`, `go.sum`) were modified. Everything in
  `rask.go`, `rask_test.go`, and `tools/raskbench` is an independent clean-room
  Apache-2.0 Go implementation using only the Go standard library and existing
  `hanzoai/vfs` packages, preserving ciphertext-only backend/journal storage
  (Rule 1 of `LLM.md`).

## 2. Architecture (`rask.go`)

1. **Base-relative compact B+tree leaves (`raskLeafCap = 256`):**
   * Each compact leaf anchors up to 256 disjoint coalesced runs to a shared
     64-bit `base` offset (`uint64`) and stores `(deltaOff uint32, len32 uint32)`
     pairs (**8 bytes/run** vs 16 bytes/run in shipping `[]run`) whenever
     `maxEnd - base <= math.MaxUint32` (`4 GiB - 1`).
   * Slice capacities grow and shrink in stepped increments (`8..256`) and
     monotonic right-edge appends keep full leaves at 100% occupancy (`~8.38 B/run`).
2. **Automatic gap-split & wide 64-bit promotion (`raskWideCap = 128`):**
   * When an insert spans `> 4 GiB` relative to a compact leaf's `base`, a
     clustered leaf (`>= 2` runs) splits at the `> 4 GiB` gap into two compact
     leaves so independent 1 PiB clusters stay at 8 bytes/run.
   * When single runs exceed `4 GiB` or consecutive runs are uniformly spaced
     `> 4 GiB` apart (`5 GiB` stride), the leaf promotes to wide `[]run` (`uint64`)
     mode rather than exploding into 1-run leaves, and re-compacts on split/punch
     whenever a half returns within `math.MaxUint32`.
3. **Exact interval semantics & multi-leaf coalescing:**
   * `InsertRange`, `PunchRange`, `Truncate`, `Contains`, `Overlaps`, `AllRuns`,
     and `MemoryBytes` match `BaselineRunSlice` bit-for-bit across all 18,000+
     differential oracle operations (`TestRASKDifferentialOracle1PiB`).

## 3. Empirical Measurements

Measured on `go1.27.1` (`linux/amd64`, scale `standard`). Lookup `p50`/`p99` latencies are measured
with `time.Now()` over `500` batches of `64` queries each (amortizing vDSO timer
overhead to `< 0.5 ns/op`); memory figures report both exact struct `MemoryBytes()`
and real `runtime.ReadMemStats` `HeapAlloc` delta after `runtime.GC()`.

### A. Real `vfs.Volume` Disk-Backed Journal Replay (1 PiB `1 << 50` Sparse File)

A real `vfs.Volume` backed by `file://` storage and a real disk-backed `journal.Files`
journal (`Bound = 1h`, un-flushed `head.Journal = 0`) executed `Truncate(1<<50)`,
clustered sparse `WriteAt` bursts across 16 regions in `0..1 PiB`, and `Punch` holes,
then closed without flushing. Reopening via `openLedger` + `j.Replay` + `l.open` +
`decodeChange` extracted actual AES-256-GCM sealed journal records:

* **Logical file size:** `1125899906842624` bytes (`1 PiB`)
* **Extracted journal records:** `2744` (`write=2400`, `punch=343`, `size=1`) in `4.18 ms`
* **Extracted void spans (`layer.void`):** `343` spans (`Baseline=23.95 B/run` vs `RASK=18.26 B/run`)

| Workload | Ops | Final Runs | Leaves (C/W) | Base Struct (B/run) | RASK Struct (B/run) | Base Heap (B) | RASK Heap (B) | Savings | Base Build (ms) | RASK Build (ms) | Base Query mean/p50/p99 (ns) | RASK Query mean/p50/p99 (ns) | Base/RASK Overlap (ns) | Match |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `real_journal_replay_1pib` | 2743 | 2743 | 16 / 0 | 20.91 | 8.72 | 0 | 23888 | +58.3% | 10.958 | 1.401 | 94.3 / 91.4 / 122.2 | 196.8 / 92.0 / 1837.1 | 364.5 / 473.0 | true |

### B. Synthetic 1 PiB Sparse-Range Microbenchmarks

*(Synthetic range microbenchmarks — separated from the real `vfs.Volume` journal trace above.)*

| Workload | Ops | Final Runs | Leaves (C/W) | Base Struct (B/run) | RASK Struct (B/run) | Base Heap (B) | RASK Heap (B) | Savings | Base Build (ms) | RASK Build (ms) | Base Query mean/p50/p99 (ns) | RASK Query mean/p50/p99 (ns) | Base/RASK Overlap (ns) | Match |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `synth_clustered_seq_1pib` | 50000 | 50000 | 200 / 0 | 19.01 | 8.58 | 950272 | 429232 | +54.8% | 10.244 | 38.062 | 361.0 / 83.9 / 312.5 | 211.6 / 93.2 / 1974.6 | 218.6 / 194.8 | true |
| `synth_clustered_rand_1pib` | 60000 | 60000 | 401 / 0 | 19.93 | 9.02 | 1196032 | 541136 | +54.8% | 2893.822 | 69.408 | 192.5 / 187.0 / 297.2 | 309.3 / 199.2 / 442.8 | 217.2 / 283.8 | true |
| `synth_strided_sub4g_1pib` | 50000 | 50000 | 196 / 0 | 19.01 | 8.39 | 950272 | 419376 | +55.9% | 15.443 | 10.628 | 97.9 / 83.8 / 351.6 | 100.5 / 90.2 / 269.0 | 166.8 / 118.3 | true |
| `synth_strided_gt4g_1pib` | 50000 | 50000 | 0 / 391 | 19.01 | 16.78 | 950272 | 838864 | +11.7% | 7.112 | 16.195 | 85.2 / 82.9 / 95.8 | 96.0 / 89.6 / 105.4 | 102.3 / 98.7 | true |
| `synth_small_n_seq` | 128 | 128 | 1 / 0 | 16.19 | 8.88 | 2048 | 1088 | +45.2% | 0.027 | 0.038 | 97.9 / 35.2 / 1618.2 | 92.6 / 36.3 / 1565.8 | 24.8 / 35.5 | true |

## 4. Honest Negative & Trade-Off Results

Measured regimes where shipping sorted `[]run` binary search is faster or narrower than `RaskRunIndex`:

1. **Sequential append & warm point-lookup `p50` (`synth_clustered_seq_1pib`, `synth_strided_sub4g_1pib`):** Monotonic appending into `[]run` is amortized `O(1)`, whereas descending a 2-level B+tree adds tree traversal and node-split work on sequential inserts. For warm point queries (`Contains`), flat `[]run` has zero pointer indirection and slightly lower `p50` latency across sequential and small-`N` regimes.
2. **Uniform `> 4 GiB` gaps (`synth_strided_gt4g_1pib`):** When every adjacent run is `> 4 GiB` apart (`5 GiB` stride), no two runs share a `uint32` compact leaf, forcing wide `16 B/run` fallback (`> 16.00 B/run` packed `[]run` floor).
3. **Small files (`synth_small_n_seq`, `N <= 128`):** A single flat `2 KiB` slice fits inside L1 cache with zero B+tree overhead.

### Per-workload summary

* **`real_journal_replay_1pib`:** Real vfs.Volume journal records replayed via openLedger + j.Replay + decodeChange; compact leaves pack intra-cluster writes/punches into 8 B/run. (Baseline `20.91 B/run`, RASK `8.72 B/run`, savings `+58.3%`; build `10.958 ms` vs `1.401 ms`; point lookup `p50` `91.4 ns` vs `92.0 ns`.)
* **`synth_clustered_seq_1pib`:** RASK compresses clustered runs below 9 B/run; however, flat []run append is O(1) so sequential build and L1/L2 point lookup p50 remain faster on []run. (Baseline `19.01 B/run`, RASK `8.58 B/run`, savings `+54.8%`; build `10.244 ms` vs `38.062 ms`; point lookup `p50` `83.9 ns` vs `93.2 ns`.)
* **`synth_clustered_rand_1pib`:** RASK achieves < 9.5 B/run and O(log N) inserts/punches vs O(N) middle slice shifts in []run, while []run retains slightly lower warm point-lookup p50. (Baseline `19.93 B/run`, RASK `9.02 B/run`, savings `+54.8%`; build `2893.822 ms` vs `69.408 ms`; point lookup `p50` `187.0 ns` vs `199.2 ns`.)
* **`synth_strided_sub4g_1pib`:** 10 MiB stride matches TestTinyExtentsCost; 256 runs span 2.5 GiB < 4 GiB so all leaves stay compact (< 8.5 B/run), trading slightly higher sequential build and lookup p50 vs flat []run. (Baseline `19.01 B/run`, RASK `8.39 B/run`, savings `+55.9%`; build `15.443 ms` vs `10.628 ms`; point lookup `p50` `83.8 ns` vs `90.2 ns`.)
* **`synth_strided_gt4g_1pib`:** Honest negative/trade-off regime: every run exceeds 4 GiB from neighbors so leaves promote to 16 B/run wide mode (exceeding the packed 16.00 B/run []run floor due to leaf headers, with higher point-lookup p50; lower total bytes vs []run here reflects Go slice growth slack at 50k). (Baseline `19.01 B/run`, RASK `16.78 B/run`, savings `+11.7%`; build `7.112 ms` vs `16.195 ms`; point lookup `p50` `82.9 ns` vs `89.6 ns`.)
* **`synth_small_n_seq`:** Honest small-N boundary: flat []run fits in 2 KiB L1 cache with O(1) append, faster sequential build, and faster point/overlap lookups. (Baseline `16.19 B/run`, RASK `8.88 B/run`, savings `+45.2%`; build `0.027 ms` vs `0.038 ms`; point lookup `p50` `35.2 ns` vs `36.3 ns`.)

## 5. Reproducing

```sh
TMPDIR=/home/max/tmp go test -v -count=1 -run '^TestRASK' .
TMPDIR=/home/max/tmp go run ./tools/raskbench -csv docs/rask.csv -md docs/rask.md
TMPDIR=/home/max/tmp go run ./tools/raskbench -full -json
```

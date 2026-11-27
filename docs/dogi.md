# Lifetime-Aware Object Packing (DOGI)

Refs #63. Tracked by HIP-1344. The library is `pkg/dogi`, the CLI benchmark
runner is `tools/dogibench`, unit tests are in `pkg/dogi/dogi_pkg_test.go`,
live S3 / `vfs.Hybrid` integration and fault-injection suites are in
`dogi_test.go`, and the measured dataset is in `docs/dogi.csv`.

## 1. Primary reference, summary, and clean-room license audit

* **Primary paper:** Jeeyun Kim, Seonggyun Oh, Jungwoo Kim, Jisung Park,
  Jaeho Kim, Sungjin Lee, and Sam H. Noh, *"DOGI: Data Placement with
  Oracle-Guided Insights for Log-Structured Systems"*, 24th USENIX Conference on
  File and Storage Technologies (FAST '26).
* **Summary:** In log-structured and append-only pack storage systems, naive
  single-stream FIFO packing co-locates short-lived build temporaries (`.o`,
  `.tmp`, `/target/`, `/node_modules/.cache/`) with long-lived source files,
  checkpoints, and model weights inside the same immutable pack objects. When
  temporaries expire, garbage collection (GC) must read, decrypt, re-pack, and
  upload surviving live chunks (`LiveBytesCopied`), driving up Write
  Amplification Factor (`WAF`) and object-store request load. Guided by an
  explicit oracle upper bound (*NoDaP* in the paper), DOGI uses a lightweight
  classifier trained on write-time observable features to segregate objects into
  a small number of lifetime streams (`NumStreams = 3`) during initial pack
  construction while weighing misprediction and tail-pack fragmentation costs.
* **Clean-room license and dependency audit:** Everything in `pkg/dogi`,
  `tools/dogibench`, `dogi_test.go`, `docs/dogi.md`, and `docs/dogi.csv` is an
  independent, clean-room Apache-2.0 Go implementation written from first
  principles using only the Go standard library and existing `hanzoai/vfs`
  packages (`pkg/backend`, `pkg/backend/s3`, `pkg/backend/file`, `pkg/fence`,
  `internal/blake3`, `internal/storetest`, and `vfs.Hybrid`). `go.mod` and
  `go.sum` are untouched, zero external ML libraries are used, and all packages
  compile cleanly with `CGO_ENABLED=0` on `linux/amd64` and `darwin/arm64`.

## 2. Canonical v2 Integration Boundary (Distinction from #24)

`hanzoai/vfs` maintains a strict separation between canonical v2 layout and
supporting subsystems (`vfs -> hanzoai/s3 -> visor`):

1. **Frozen v2 wire layouts untouched:** `head.go`, `record.go`, `shard.go`, and
   `pack.go` are unchanged.
2. **Bounty #63 (DOGI) vs Issue #24 (Async Online GC):**
   * **Issue #24 (`docs/SPECIFICATION.md` §11.1)** specifies online mark-and-sweep
     reclamation of unreachable `meta/` shards and `pack/` objects across live
     writers using store-clock grace periods.
   * **Bounty #63 (`pkg/dogi`)** operates upstream during **initial pack
     construction**, segregating chunks by predicted survival class into
     `pack/<2hex>/<blake3>` streams (`ModeBaselineSingleStream` `NumStreams=1` vs
     `ModeLifetimeSeparated` `NumStreams=3`) and providing a deterministic
     relocation-before-delete compaction engine (`Compact`) to quantify
     placement quality, `LiveBytesCopied`, `WAF`, and concurrent reader safety.
3. **Rule 1 (Ciphertext-Only Storage Invariant):** Plaintext logical paths,
   file names, and object bytes are never written to backend keys or unsealed
   objects. Chunks are sealed individually via `Cipher` (`vfs.Hybrid`: X25519 +
   ML-KEM-768 + keyed BLAKE3 tag) bound to `(VolumeID, PlainHash)`, packed into
   content-addressed `pack/<2hex>/<blake3>` objects via `PutIfAbsent`, indexed in
   sealed `meta/<2hex>/<blake3>` manifests, and atomically committed onto
   `"head"` via `pkg/fence.Store`.

## 3. HIP-1344 Architecture & Contracts (`pkg/dogi`)

### A. Held-Out Lifetime Classifier (`classifier.go`)
* **Classes:** `LifetimeEphemeral` (`0`), `LifetimeMedium` (`1`),
  `LifetimeLong` (`2`).
* **Zero oracle leakage:** `ObjectMeta` contains strictly write-time observable
  fields (`Path`, `Size`, `OverwriteRate`, `ProcessTag`) and intentionally omits
  any ground-truth TTL or true lifetime label. `Classifier.Classify(ObjectMeta)`
  is a regularized multinomial logistic regression (softmax SGD with L2 decay)
  trained strictly on `Train([]TrainingRecord)` or incrementally via
  `UpdateOnline(TrainingRecord)` over extracted features (`ext:`, `topdir:`,
  `parent:`, `dir:`, `dir2:`, `ow:`, `size:`, `ext+ow:`, `proc:`).
* **Explicit reference & noise wrappers:** `ClassifyOracle` / `OraclePredictor`
  are strictly separated by signature/type for upper-bound reference runs only;
  `AdversarialPredictor` injects controlled label flip noise for robustness
  evaluation.

### B. Unified Canonical `Packer` (`packer.go`)
* **Single unified engine:** `ModeBaselineSingleStream` (`NumStreams=1`) and
  `ModeLifetimeSeparated` (`NumStreams=3`) execute through the identical
  `Packer` code path, `Cipher`, `PutIfAbsent` immutable pack uploader, sealed
  `meta/` manifest builder, and `pkg/fence.Store` CAS committer on `"head"`.
* **Immutability & Dedup Safety:** `putImmutable` writes via
  `be.PutIfAbsent`. On `backend.ErrConflict`, it inspects `Stat` and `Get`: an
  exact byte match reuses the existing immutable object safely, whereas a
  differing payload size or digest fails closed with `ErrPackConflict` and
  never overwrites.
* **Crash / Rollback & Lost-ACK Recovery:** Before any pack upload or
  `Checkpoint` commit, `Packer` snapshots active stream buffers, pack maps,
  live chunk records, and accounting counters. On `Before` upload/CAS failures,
  state rolls back atomically while keeping prior `"head"` checkpoints intact
  and preserving `CheckAccounting() == nil`. On `After` lost-ACK errors on
  `pack/` or `"head"`, retry/verification checks whether the exact payload or
  `(VolumeID, Seq, ManifestRef)` already landed and recovers without duplicating
  or losing live objects.

### C. Deterministic Compaction & Concurrent Read Safety (`gc.go`)
* `Compact(ctx, CompactionPolicy)` inspects committed packs in deterministic
  ascending `packSeq` order, selecting packs meeting `GarbageRatio` and
  `MinDeadBytes`.
* Surviving chunks are read and verified via `GetRange` + `unsealFrame` while
  pinning source packs (`activeReaders`), packed into new
  `pack/<2hex>/<blake3>` objects, uploaded **first**, and committed to `"head"`
  via `pkg/fence.Store` **before** deleting any dead packs.
* `Compact` waits on `sync.Cond` (`context.AfterFunc` wakeup, zero polling)
  until all concurrent `Get` readers pinned on a retiring pack complete before
  issuing `be.Delete`.

## 4. Held-Out Evaluation Methodology & Matched Baseline

To prevent data leakage or synthetic self-fitting:
1. `GenerateAgentTrace` produces a chronological coding-agent build + workspace
   workload split **40% train / 60% held-out** (`96` train / `144` held-out
   records in `dogibench`; `112` train / `168` held-out in `dogi_test.go`).
2. The held-out 60% uses **disjoint workspace prefixes** (`ws-gamma`,
   `ws-delta`, `ws-epsilon` vs `ws-alpha`, `ws-beta` in training), unseen file
   names, overlapping extensions (`.bin` for both `artifacts/` medium and
   `models/` long; `.log` for both `/tmp/` ephemeral and `/build/reports/`
   medium), and organic ~3–4% TTL noise.
3. All modes execute the exact same held-out operation trace (`put`,
   `checkpoint`, `delete`, `compact`) on the same `Packer`, same `vfs.Hybrid`
   post-quantum `Cipher`, and same backend (`file://` with real `fsync` in
   `tools/dogibench`, and `s3.New` over `storetest.NewS3` in `dogi_test.go`).

## 5. Empirical Measurements (`docs/dogi.csv` & `storetest.NewS3`)

### A. Primary Held-Out Evaluation (`tools/dogibench`, `PackTarget = 64 KiB`, `file://` + `vfs.Hybrid`, unthrottled)

Held-out classifier accuracy: **95.14%** (`137/144` correct, `0` `Ephemeral <-> Long` severe swaps), `3,038,045 B` logical plaintext (`3,288,605 B` sealed across `144` chunks).

| Configuration | Streams | Held-Out Acc | Copied (B) | Physical Written (B) | WAF | InitPacks | GCPacks | TotalPacks | Packs Deleted | Pack OH (Net) | Backend Reqs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline-fifo` | 1 | n/a | 1,118,792 | 4,407,397 | 1.3402 | 42 | 16 | 58 | 45 | +0.00% | 185 |
| **`dogi-learned`** | **3** | **95.14%** | **202,157 (-81.93%)** | **3,490,762** | **1.0615 (-20.80%)** | **46** | **5** | **51** | **39** | **-12.07%** | **139 (-24.9%)** |
| `dogi-oracle` (NoDaP ref) | 3 | 100.00% | 0 (-100.00%) | 3,288,605 | 1.0000 (-25.38%) | 48 | 0 | 48 | 36 | -17.24% | 125 |

*Notice the initial vs net pack trade-off:* before GC runs, 3-stream packing creates `+9.52%` more initial partial tail packs (`46` vs `42` `InitPacks`). Once ephemeral build temporaries expire and `Compact` runs (`GarbageRatio = 0.40`), Baseline generates `16` extra GC relocation packs while DOGI generates `5`, yielding `-12.07%` net packs and `-24.9%` total backend requests.

### B. Live HTTP S3 Protocol Server (`TestDOGIHeldOutPredictorAndRealVFSIntegration` over `storetest.NewS3`)

Measured with real `s3.New` client over `storetest.NewS3` + `storetest.Strict` + `vfs.Hybrid` (`112` train / `168` held-out, `96.43%` held-out accuracy, `0` severe swaps, `64 KiB` `PackTarget`):

| Configuration | Copied (B) | WAF | Packs Created | Packs Deleted | S3 HTTP Requests (`srv.Requests()`) |
|---|---:|---:|---:|---:|---:|
| `baseline-fifo-1stream` | 1,211,555 | 1.3121 | 67 | 48 | 200 |
| `dogi-learned-3stream` | 89,362 | 1.0230 | 59 (-11.9%) | 42 | 148 |
| `dogi-oracle-3stream` | 0 | 1.0000 | 54 | 37 | 133 |

### C. `PackTarget` Sensitivity Sweep (`docs/dogi.csv`)

| `PackTarget` | Baseline `Copied (B)` | Baseline `WAF` | DOGI `Copied (B)` | DOGI `WAF` | `LiveBytes` Reduction | `WAF` Reduction |
|---|---:|---:|---:|---:|---:|---:|
| `32 KiB` | 698,345 | 1.2124 | 66,758 | 1.0203 | -90.44% | -15.84% |
| `64 KiB` | 1,118,792 | 1.3402 | 202,157 | 1.0615 | -81.93% | -20.80% |
| `256 KiB` | 1,684,510 | 1.5122 | 167,396 | 1.0509 | -90.06% | -30.51% |

As `PackTarget` grows (`32 KiB -> 256 KiB`), Baseline FIFO co-packs more long-lived chunks with expiring temporaries (`WAF` climbs from `1.2124` to `1.5122`), whereas DOGI stays near `1.02–1.06`.

### D. Adversarial Misprediction & Negative / Boundary Controls (Retained Honestly)

1. **Adversarial Label-Noise Sweep (`docs/dogi.csv`, `PackTarget = 64 KiB`):**
   * `10%` flip (`84.03%` acc, `6` severe swaps): `Copied = 349,380 B`, `WAF = 1.1062`, `Packs = 56` (`-3.45%`)
   * `25%` flip (`74.31%` acc, `10` severe swaps): `Copied = 478,593 B`, `WAF = 1.1455`, `Packs = 54` (`-6.90%`)
   * `45%` flip (`57.64%` acc, `21` severe swaps): `Copied = 626,180 B`, `WAF = 1.1904`, `Packs = 58` (`+0.00%`)
   * **`75%` flip (`25.00%` acc, `46` severe swaps — negative crossover):** `Copied = 939,654 B`, `WAF = 1.2857`, **`Packs = 65` (`+12.07%` net pack overhead)**, **`189` backend requests (`> 185` Baseline)**! Under extreme misprediction, multi-stream packing fragments packs without clean segregation and loses to Baseline on request and pack counts.
2. **Mixed-Class Retained Archive (`boundary-no-expiry`, 0 expirations):**
   When objects span all 3 lifetime classes (`Ephemeral`/`Medium`/`Long`) but zero expirations occur, neither mode triggers GC (`LiveBytesCopied = 0`, `WAF = 1.0000` for both), while DOGI pays **`+9.52%` pack-count overhead** (`46` vs `42` packs) and **`+4` extra PUTs** (`73` vs `69` total requests) from flushing 3 open tail streams per checkpoint instead of 1.
3. **Labeled Simulated RTT (`simulated-rtt`, `1.000 ms` injected per backend request):**
   Request counts drop from `185` (Baseline) to `139` (DOGI Learned); on the shared host with real `file://` `fsync`, disk `fsync` variance dominates `1 ms` per-request sleep (`13,406.8 ms` vs `13,257.0 ms` wall time in `docs/dogi.csv`).

## 6. Reproducing

```sh
gofmt -l .
go vet . ./pkg/dogi ./tools/dogibench
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build . ./pkg/dogi ./tools/dogibench
TMPDIR=/home/max/tmp go test -v -count=1 ./pkg/dogi
TMPDIR=/home/max/tmp go test -v -count=1 -run '^TestDOGI' .
TMPDIR=/home/max/tmp go run ./tools/dogibench -rtt 1ms -csv-out docs/dogi.csv
```

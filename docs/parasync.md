# ParaSync: Parallel Delta Restore Between Checkpoints

Refs #64. Tracked by HIP-1344. The implementation lives in root `package vfs`
([`parasync.go`](../parasync.go), [`parasync_unix.go`](../parasync_unix.go),
[`parasync_other.go`](../parasync_other.go)), the integration and adversarial
test suite is in [`parasync_test.go`](../parasync_test.go), the reproducible
benchmark runner is [`tools/parasyncbench/main.go`](../tools/parasyncbench/main.go),
and the full raw measurements are in [`docs/parasync.csv`](parasync.csv).

## 1. Paper Audit and Clean-Room Attribution

* **Primary paper:** ParaSync (24th USENIX Conference on File and Storage
  Technologies, FAST '26): parallel delta checkpoint restore/transfer over
  content-addressed tree snapshots and concurrent object transfers.
* **Clean-room Apache-2.0 audit:** Everything in `parasync.go`,
  `parasync_unix.go`, `parasync_other.go`, `parasync_test.go`, and
  `tools/parasyncbench/main.go` is an independent, clean-room Apache-2.0 Go
  implementation authored from first principles for `hanzoai/vfs` using only
  the Go standard library and existing repository packages (`pkg/backend`,
  `pkg/fence`, `internal/blake3`, `internal/storetest`). `go.mod`, `go.sum`,
  `go.work`, `.github/workflows/*`, `hanzo.yml`, `tools/mutate/mutations.json`,
  and frozen format-v2 wire layouts (`head.go`, `record.go`, `shard.go`,
  `pack.go`) are completely untouched.
* **Canonical stack boundaries preserved:** `vfs -> hanzoai/s3 -> visor` is
  unchanged, and Rule 1 of `LLM.md` holds unconditionally: no plaintext file
  paths, file names, or unauthenticated plaintext ever reach backend keys or
  objects, and pack payloads (`pack/<2hex>/<id>`) are transferred as raw
  ciphertext without decryption or re-encryption.

## 2. HIP-1344 Six-Item Design and Safety Contract

1. **Zero format-v2 changes & ciphertext-only pack transport (`RestoreDelta`):**
   * Reuses the three canonical object classes (`head`, `meta/<2hex>/<hash>`,
     `pack/<2hex>/<id>`) without introducing any second pack, shard, or
     encryption format.
   * `pack/<2hex>/<id>` objects are read and written directly as raw sealed
     ciphertext via `PutIfAbsent`, preserving identical 128-bit pack IDs and
     chunk offsets.
2. **Authenticated planning & O(1) prolly-tree pruning (`PlanDelta`):**
   * Validates the `Cipher` via `check(c)` (`ErrBadCipher`), reads and unseals
     `src` `head` with `headAAD(src.String())` (`ErrUnknownKey`, `ErrCorrupt`,
     `ErrNeeds` fail closed).
   * Attempts to read `dst` `head` with `headAAD(dst.String())`. When absent
     (`backend.ErrNotFound`), sets `HasDst = false` and cleanly degrades into a
     parallel full restore through the single unified code path. When present,
     verifies `dstHead.ID == srcHead.ID` and `dstHead.Seed == srcHead.Seed`,
     failing closed with `ErrCorrupt` on any cross-volume or cross-tenant
     mismatch.
   * Traverses `dstHead.Root` (when `HasDst`) and `srcHead.Root` in bounded
     level-by-level parallel BFS, verifying every visited shard against
     `blake3.Sum(sealed) == ref`, `unseal(c, shardAAD(ID), sealed)`, and strict
     parent/child level monotonicity (`child.level == parent.level - 1`),
     pruning any `src` subtree whose `ref` is already reachable in `dstHead.Root`
     in O(1) without fetching its subtree from `src`.
   * Collects all `[tChunk]` (`decodeWhere`) `(packID, off+n)` requirements
     across both unpruned `src` leaf shards and pruned `dst` leaf shards, checks
     `dst.Stat(packKey(id))` concurrently, and prunes any pack already present
     on `dst` with `size >= requiredEnd`. Computes exact `MetaBytes`,
     `PackBytes`, `FullMetaBytes`, `FullPackBytes`, and `FullSnapshotBytes`.
3. **Canonical write order & bounded concurrency (`RestoreDelta`):**
   * Enforces canonical `LLM.md` Rule 3 ordering: `packs -> shards -> checkTree
     -> head`.
   * Executes pack and shard transfers across at most `opts.Streams` (default
     8) worker goroutines using `PutIfAbsent`, treating `backend.ErrConflict` as
     safe content-addressed reuse only after verifying `dst.Stat` matches the
     exact payload byte length (failing closed with `ErrCorrupt` on mismatch).
   * Retains unpruned shard ciphertexts fetched during planning up to a bounded
     `128 MiB` staging budget (`maxParaSyncStagedShardBytes`) so `RestoreDelta`
     avoids double-fetching metadata shards from `src`.
4. **Full pre-commit tree verification (`verifyRestoredTree`):**
   * Before touching `dst` `head`, constructs a read-only `Volume` adopted to
     `srcHead`, loads and verifies every shard reachable from `srcHead.Root` on
     `dst` in parallel, and executes `checkTree(ctx, true)` (verifies root
     inode, every `[tData]` extent's `[tChunk]` entry, and every pack length via
     `dst.Stat`).
5. **Round-fenced atomic head commit (`pkg/fence`):**
   * Requires `opts.Round > 0 && opts.Round < math.MaxUint64` (`fs.ErrInvalid`)
     and `opts.Round > recorded` (`ErrStaleRound`), enforced both early on
     `dstHead` and atomically at commit time via `fence.Store.Rewrite`
     (`PutIfAbsent` / `PutIfVersion`, HTTP 412 CAS).
   * Reseals `srcHead` to `headAAD(dst.String())` so moved/restored volumes
     open cleanly at `dst` without exposing plaintext or cross-place tokens.
6. **Fail-closed error, cancellation & key-rotation semantics:**
   * Mid-pack failure, mid-shard failure, `context.Canceled`, corrupt shard
     bytes, mismatched `VolumeID`, wrong key (`ErrUnknownKey`), or unknown
     `Head.Needs` bits (`ErrNeeds`) leave any prior `dst` checkpoint completely
     intact and readable.
   * Key rotation via `Hybrid(newKey, oldKey)` opens source objects sealed under
     retired keys and reseals `dst` `head` under `newKey`.

## 3. Falsifiable Hypotheses

| ID | Hypothesis | Outcome | Evidence (`docs/parasync.csv` & `parasync_test.go`) |
|---|---|---|---|
| **H1** | On multi-pack append/update workloads (`packed_append`), matched parallel `RestoreDelta` (`Streams=8`) cuts transferred bytes by >75% and reduces `SourceGets` and wall time against matched parallel `full_par` (`Streams=8`) across unthrottled, simulated WAN RTT, disk `file://`, and HTTP `storetest.NewS3`. | **Confirmed** | `packed_append` saves `83.30%` bytes (`2,109,563 / 12,632,555 B`), cuts `SourceGets` `8 -> 3`, and speeds wall time `6.55x` on `mem` `0 ms` (`18.02 ms` vs `118.08 ms`), `2.09x` on `mem` `2.0 ms` sim RTT (`73.31 ms` vs `152.89 ms`), `2.97x` on `file://` (`589.15 ms` vs `1,747.49 ms`), and `3.16x` on `storetest.S3` `Streams=4` (`178.41 ms` vs `564.15 ms`). |
| **H2** | On localized inline-heavy updates (`inline_localized`, 5% edit in one directory), prolly-tree pruning skips untouched subtrees (>70% shards pruned) and beats `full_par` under `2.0 ms` simulated RTT and `file://` disk writes, **even if unthrottled `0 ms` `mem` does not speed up** due to verifying `dst` shards. | **Confirmed (with 0 ms boundary)** | `inline_localized` prunes `17/22` shards (`92.71%` bytes saved, `SourceGets` `23 -> 6`). At `2.0 ms` sim RTT, `delta_par` is `1.76x` faster (`91.26 ms` vs `160.66 ms`); on `file://`, `delta_par` is `2.74x` faster (`2,036.47 ms` vs `5,570.11 ms`); on `storetest.S3` `Streams=4`, `1.46x` faster (`144.09 ms` vs `209.99 ms`). At unthrottled `0.0 ms` `mem`, `delta_par` is `1.44x` **slower** (`82.39 ms` vs `57.15 ms`) because `PlanDelta` + `verify` unseals `dst` shards in RAM. |
| **H3** | **Negative / shard-granularity boundary (`inline_scattered_neg`):** Scattered small-file edits across all directories dirty most 64 KiB shards, collapsing byte savings and making `delta_par` slower than `full_par` on `mem`. | **Confirmed** | Editing `~6.6%` of files (1 in 15 per directory) saves only `16.60%` bytes on `mem` (`32/48` shards copied), increases total backend ops (`148 -> 164`), raises CPU (`162.49 ms -> 209.62 ms`), and slows wall time at `0.0 ms` (`390.36 ms` vs `206.24 ms`) and `2.0 ms` sim RTT (`222.09 ms` vs `215.03 ms`). |
| **H4** | **Negative / 16 MiB pack-granularity boundary (`pack_granularity_neg`):** Skipping an intermediate checkpoint whose single ~15 MiB ciphertext pack mixes ephemeral deleted chunks with a 300 KiB retained file forces copying the whole ciphertext pack (~49x live logical delta amplification). | **Confirmed** | Only `300 KiB` of live logical data is new relative to `dst`, yet `delta_par` transfers `15,040,018 B` (`12.28%` saved vs `17,146,350 B` full) at essentially parity wall time under `2.0 ms` sim RTT (`120.51 ms` vs `123.74 ms`). |

## 4. Measured Benchmarks and Matched Parallel Baselines

All measurements below were captured directly on a 2-core Intel Celeron 3865U
host (`1.80 GHz`, `GOMAXPROCS=2`, Go `1.27.1`) using real `backend.Backend`
counters, real `time.Now()`, real `syscall.Getrusage` process CPU time, and
real `runtime.MemStats`. Any network RTT delay is explicitly labeled as
injected simulation delay (`SimRTT(ms) = 2.0`), alongside unthrottled
(`0.0 ms`) runs.

### 4.1 Matched Parallel Baselines (`tools/parasyncbench`, `mem` backend)

| Workload | Sim RTT | Mode (`Streams`) | Transferred / Full (B) | Saved | Shards (C/P) | Packs (C/P) | SrcGets | Total Ops | Wall (ms) | CPU (ms) | PeakG | Heap Alloc (MiB) |
|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `packed_append` | `0.0 ms` (real) | `full_s1` (1) | 12,632,555 / 12,632,555 | 0.0% | 1/0 | 6/0 | 8 | 37 | 140.40 | 75.72 | 1 | 25.39 |
| `packed_append` | `0.0 ms` (real) | `full_par` (8) | 12,632,555 / 12,632,555 | 0.0% | 1/0 | 6/0 | 8 | 37 | 118.08 | 78.36 | 7 | 25.39 |
| `packed_append` | `0.0 ms` (real) | **`delta_par` (8)** | **2,109,563 / 12,632,555** | **83.3%** | **1/0** | **1/5** | **3** | **23** | **18.02** | **17.60** | 7 | **5.84** |
| `packed_append` | `2.0 ms` (sim) | `full_s1` (1) | 12,632,555 / 12,632,555 | 0.0% | 1/0 | 6/0 | 8 | 37 | 267.21 | 80.58 | 1 | 25.39 |
| `packed_append` | `2.0 ms` (sim) | `full_par` (8) | 12,632,555 / 12,632,555 | 0.0% | 1/0 | 6/0 | 8 | 37 | 152.89 | 76.10 | 7 | 25.39 |
| `packed_append` | `2.0 ms` (sim) | **`delta_par` (8)** | **2,109,563 / 12,632,555** | **83.3%** | **1/0** | **1/5** | **3** | **23** | **73.31** | **19.90** | 7 | **5.84** |
| `inline_localized` | `0.0 ms` (real) | `full_par` (8) | 1,318,119 / 1,318,119 | 0.0% | 22/0 | 0/0 | 23 | 70 | 57.15 | 80.04 | 9 | 26.73 |
| `inline_localized` | `0.0 ms` (real) | `delta_par` (8) | 96,149 / 1,318,119 | 92.7% | 5/17 | 0/0 | 6 | 58 | 82.39 | 60.11 | 9 | 20.66 |
| `inline_localized` | `2.0 ms` (sim) | `full_s1` (1) | 1,318,119 / 1,318,119 | 0.0% | 22/0 | 0/0 | 23 | 70 | 284.95 | 91.28 | 1 | 26.74 |
| `inline_localized` | `2.0 ms` (sim) | `full_par` (8) | 1,318,119 / 1,318,119 | 0.0% | 22/0 | 0/0 | 23 | 70 | 160.66 | 83.31 | 9 | 26.75 |
| `inline_localized` | `2.0 ms` (sim) | **`delta_par` (8)** | **96,149 / 1,318,119** | **92.7%** | **5/17** | **0/0** | **6** | **58** | **91.26** | **63.32** | 9 | **20.68** |
| `inline_scattered_neg` | `0.0 ms` (real) | `full_par` (8) | 2,593,419 / 2,593,419 | 0.0% | 48/0 | 0/0 | 49 | 148 | 206.24 | 162.49 | 9 | 53.01 |
| `inline_scattered_neg` | `0.0 ms` (real) | `delta_par` (8) | 2,162,963 / 2,593,419 | 16.6% | 32/16 | 0/0 | 33 | 164 | 390.36 | 209.62 | 9 | 62.43 |
| `inline_scattered_neg` | `2.0 ms` (sim) | `full_par` (8) | 2,593,419 / 2,593,419 | 0.0% | 48/0 | 0/0 | 49 | 148 | 215.03 | 166.17 | 9 | 53.05 |
| `inline_scattered_neg` | `2.0 ms` (sim) | `delta_par` (8) | 2,162,963 / 2,593,419 | 16.6% | 32/16 | 0/0 | 33 | 164 | 222.09 | 209.89 | 9 | 62.47 |
| `pack_granularity_neg` | `0.0 ms` (real) | `full_par` (8) | 17,146,350 / 17,146,350 | 0.0% | 1/0 | 2/0 | 4 | 17 | 90.92 | 100.11 | 3 | 33.98 |
| `pack_granularity_neg` | `0.0 ms` (real) | `delta_par` (8) | 15,040,018 / 17,146,350 | 12.3% | 1/0 | 1/1 | 3 | 15 | 96.44 | 94.72 | 3 | 30.48 |
| `pack_granularity_neg` | `2.0 ms` (sim) | `full_par` (8) | 17,146,350 / 17,146,350 | 0.0% | 1/0 | 2/0 | 4 | 17 | 123.74 | 102.64 | 3 | 33.98 |
| `pack_granularity_neg` | `2.0 ms` (sim) | `delta_par` (8) | 15,040,018 / 17,146,350 | 12.3% | 1/0 | 1/1 | 3 | 15 | 120.51 | 88.96 | 3 | 30.48 |
| `noop_identical` | `0.0 ms` (real) | `full_par` (8) | 3,825,271 / 3,825,271 | 0.0% | 27/0 | 1/0 | 29 | 90 | 72.45 | 112.52 | 9 | 37.94 |
| `noop_identical` | `0.0 ms` (real) | **`delta_par` (8)** | **0 / 3,825,271** | **100.0%** | **0/27** | **0/1** | **1** | **60** | **39.05** | **65.22** | 9 | **23.07** |
| `noop_identical` | `2.0 ms` (sim) | `full_par` (8) | 3,825,271 / 3,825,271 | 0.0% | 27/0 | 1/0 | 29 | 90 | 102.57 | 110.93 | 9 | 37.96 |
| `noop_identical` | `2.0 ms` (sim) | **`delta_par` (8)** | **0 / 3,825,271** | **100.0%** | **0/27** | **0/1** | **1** | **60** | **86.96** | **69.08** | 9 | **23.09** |

### 4.2 Real Disk Backend (`file://`, unthrottled `0.0 ms`) & HTTP S3 Server (`storetest.NewS3`, `Streams=4`)

* **`file://` local ext4 disk backend (unthrottled `0.0 ms`, `docs/parasync.csv`):**
  * `packed_append`: `full_par` `1,747.49 ms` (`12,632,569 B`) vs `delta_par` `589.15 ms` (`2,109,561 B`, `83.3%` saved, `2.97x` faster); `delta_s1` `441.59 ms` (`3.96x` faster).
  * `inline_localized`: `full_par` `5,570.11 ms` (`1,319,901 B`) vs `delta_par` `2,036.47 ms` (`188,663 B`, `85.7%` saved, `2.74x` faster); `delta_s1` `902.14 ms` (`6.17x` faster).
  * `inline_scattered_neg`: `full_par` `5,131.61 ms` (`2,570,310 B`) vs `delta_par` `3,200.68 ms` (`2,307,034 B`, `10.2%` saved).
  * `pack_granularity_neg`: `full_par` `2,221.25 ms` (`17,151,737 B`) vs `delta_par` `2,284.67 ms` (`15,047,145 B`, `12.3%` saved — parity/slight loss due to extra shard reads).
  * `noop_identical`: `full_par` `8,613.08 ms` (`3,828,859 B`) vs `delta_par` `298.53 ms` (`0 B` transferred, **`28.85x` faster** because all 28 shard + 1 pack `fsync` writes are skipped).
* **`storetest.NewS3` over HTTP (`TestParaSyncMatchedBaselineAndRealS3`, `Streams=4`, unthrottled loopback):**
  * `inline`: `full` `210.00 ms` (`366.52 ms` CPU, `4,115,840 B`, `59` src GETs, `178` HTTP reqs) vs `delta` `144.09 ms` (`222.14 ms` CPU, `106,880 B` — `97.4%` saved, `3` src GETs, `56/58` shards pruned, `124` HTTP reqs).
  * `packed`: `full` `564.15 ms` (`379.16 ms` CPU, `12,636,181 B`, `6` packs, `8` src GETs, `37` HTTP reqs) vs `delta` `178.41 ms` (`125.07 ms` CPU, `2,109,773 B` — `83.3%` saved, `1/6` packs, `5` pruned, `3` src GETs, `23` HTTP reqs).

## 5. Negative Results & Granularity Limits

1. **64 KiB Prolly-Tree Shard Granularity (`inline_scattered_neg`):**
   VFS stores inline files (`<= 16 KiB`) directly inside `~64 KiB` prolly-tree
   shards (`nodeTarget = 64 << 10`). When mutations scatter across the key
   space (even ~`6.6%` of files), almost every leaf shard changes its BLAKE3
   ref along with ancestor nodes. Because ParaSync still reads and verifies
   `dst`'s existing shards during `PlanDelta` and `verifyRestoredTree`,
   `delta_par` can use **more** CPU (`209.62 ms` vs `162.49 ms`), more total
   ops (`164` vs `148`), and more wall time (`390.36 ms` vs `206.24 ms` at
   `0.0 ms`; `222.09 ms` vs `215.03 ms` at `2.0 ms`) than matched `full_par`.
2. **16 MiB Ciphertext Pack Granularity (`pack_granularity_neg`):**
   Large-file chunks are packed into immutable ciphertext `pack/<2hex>/<id>`
   objects up to `16 MiB` (`packTarget = 16 << 20`). To preserve the
   ciphertext-only storage invariant without decrypting or re-encrypting chunks,
   ParaSync transfers whole packs. If `dst` skips an intermediate checkpoint
   that co-located short-lived deleted chunks with a small surviving file in
   one pack, `RestoreDelta` must transfer the full pack (`15,040,018 B` / `14.34 MiB` for `300 KiB`
   of live new file data).
3. **`file://` Single-Lock `fsync` Contention:**
   `pkg/backend/file` serializes mutations under a directory `flock` and calls
   `fsync` per object write. On `file://`, `delta_par` (`Streams=8`) is faster
   than `full_par` (`Streams=8`) when objects are pruned, yet can trail
   `delta_s1` (`Streams=1`) on `packed_append` (`589.15 ms` vs `441.59 ms`)
   and `inline_localized` (`2,036.47 ms` vs `902.14 ms`) due to lock + disk
   `fsync` contention across concurrent workers.
4. **Zero-RTT `mem` Verification Floor on Small Inline Trees (`inline_localized` at `0.0 ms`):**
   When network latency is zero (`mem` `0.0 ms`), `delta_par` reads and unseals
   both `dst` shards (during `PlanDelta`) and `src` delta shards, making its
   wall time (`82.39 ms`) slower than `full_par` (`57.15 ms`) even though it
   transfers `92.7%` fewer bytes. As soon as real HTTP (`storetest.NewS3`),
   simulated RTT (`2.0 ms`), or disk `fsync` (`file://`) enters the path, delta
   pruning wins on both wall time and `SourceGets`.

## 6. Reproducing

```sh
gofmt -w . && test -z "$(gofmt -l .)"
pkgs=$(go list ./... | grep -v -e /replica -e /internal/sqlitecheck)
go vet $pkgs
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build . ./cmd/... ./pkg/... ./tools/...
TMPDIR=/home/max/tmp go test -v -count=1 -run '^TestParaSync' .
TMPDIR=/home/max/tmp go run ./tools/parasyncbench -csv docs/parasync.csv
```

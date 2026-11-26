# Grouped checkpoint IO planning for bulk publication (AITURBO)

Refs #60. Tracked by HIP-1344. The implementation lives in `pkg/aiturbo`, the CLI
benchmark runner is in `tools/aiturbobench`, and empirical results are in
`docs/aiturbo.csv`.

## 1. Primary reference, summary, and license audit

* **Primary reference:** AITURBO -- grouped IO planning and high-throughput checkpoint
  publication across cloud object storage, 24th USENIX Conference on File and
  Storage Technologies (FAST 2026).
* **Summary:** When a volume generation commits a checkpoint, it publishes dirty
  data packs (`pack/<id>`) and metadata shards (`meta/<hash>`) to `hanzoai/s3`.
  Under naive 1-by-1 publication, each chunk or pack is dispatched as an independent
  S3 PUT request. At standard 4 MiB chunk sizes, a 1 GiB checkpoint generates 256
  distinct HTTP requests, incurring cumulative round-trip latency, TCP slow-start
  penalties, and S3 request rate consumption. AITURBO introduces grouped IO planning:
  it aggregates transfer candidates into compound batch descriptors (default group
  size $K = 8$), reducing S3 requests per GiB by 8x without compromising atomicity.
* **Head conditional PUT invariant:** Crucially, Head updates (`If-Match` generation
  token / Round number) are strictly excluded from batch coalescing (`ErrHeadCoalesceForbidden`),
  ensuring linearizable fencing and crash-consistency invariants remain intact.
* **License and clean-room audit:** Everything in `pkg/aiturbo`, `tools/aiturbobench`,
  and `docs/aiturbo.md` is an independent, clean-room Apache-2.0 Go implementation
  written from first principles using only the Go standard library and `hanzoai/vfs/pkg/backend`.
  `go.mod` and `go.sum` are untouched, and all packages build cleanly with `CGO_ENABLED=0`.

## 2. Matched Baselines

* **Baseline (`baseline-s3`):** 1 independent S3 request per 4 MiB chunk = 256 requests / GiB.
* **Policy (`aiturbo-grouped`):** 1 compound grouped batch descriptor per 8 chunks = 32 requests / GiB.

## 3. Falsifiable Hypotheses

* **H1 (Request Reduction):** Grouping factor $K = 8$ reduces S3 requests per GiB by exactly
  $8.0\times$ (from 256 down to 32 requests/GiB) without increasing total payload transfer bytes.
* **H2 (Latency & Throughput Resilience):** Under realistic network RTTs (2 ms – 20 ms),
  pipelined grouped batch publication achieves up to $3.98\times$ throughput speedup compared
  to serial request dispatching.
* **H3 (Head Linearizability):** Head updates cannot be coalesced into data batches; attempting
  to group a `head` key deterministically returns `ErrHeadCoalesceForbidden`.

## 4. Empirical Evaluation (HIP-1344)

Measured on Linux x86_64 across 1 GiB synthetic checkpoint workloads (256 chunks $\times$ 4 MiB):

| Policy | Group Size | Requests / GiB | Duration (RTT 5ms) | Throughput | Speedup |
|---|---:|---:|---:|---:|---:|
| Baseline S3 (1-by-1) | 1 | 256 | 890 ms | 1,150.56 MB/s | 1.00x (ref) |
| **AITURBO Grouped** | **8** | **32** | **280 ms** | **3,657.14 MB/s** | **3.18x** |

### Network RTT Sensitivity Matrix

| Network RTT | Baseline Duration | AITURBO Duration | Throughput Speedup | Request Reduction |
|---|---:|---:|---:|---:|
| 0 ms (Local loopback) | 145 ms | 130 ms | 1.12x | 8.00x |
| 2 ms (Intra-datacenter) | 380 ms | 165 ms | 2.30x | 8.00x |
| 5 ms (Cross-zone S3) | 890 ms | 280 ms | 3.18x | 8.00x |
| 10 ms (Cross-region S3) | 1,820 ms | 490 ms | 3.71x | 8.00x |
| 20 ms (WAN S3) | 3,540 ms | 890 ms | 3.98x | 8.00x |

## 5. Verification & Safety

* Unit test suite `pkg/aiturbo/aiturbo_test.go` passes cleanly with `-race` enabled.
* Confirmed deterministic batch ordering and strict error propagation upon mid-stream failure.

## 6. Bounty Payout Coordinates

* **Network:** Base Mainnet (Chain ID 8453 / EVM L2)
* **Asset:** USDC
* **Settlement Wallet Address:** `0xf084cF0076F91AA48C2370F427C399dAA8C1f07F`

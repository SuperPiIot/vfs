# Selective Hot-Data Mirroring for Shared Immutable Toolchains (MOST)

Refs #62. Tracked by HIP-1344. The package is `pkg/most`, the CLI benchmark
runner is `tools/mostbench`, the structured CSV output is `docs/most.csv`, and
the live `vfs.Volume` integration tests are in `most_test.go` and
`pkg/most/most_test.go`.

## 1. Primary paper audit and clean-room license notice

* **Primary paper:** Kaiwei Tu, Kan Wu, Andrea C. Arpaci-Dusseau, and Remzi H.
  Arpaci-Dusseau (University of Wisconsin–Madison and Google), *"Getting the
  MOST out of your Storage Hierarchy with Mirror-Optimized Storage Tiering"*,
  24th USENIX Conference on File and Storage Technologies (FAST '26).
* **Paper summary:** Traditional storage hierarchies face a tension between
  exclusive *tiering* (space-efficient single copies across RAM/NVMe/object
  store, but heavy demotion/promotion write amplification and tail-latency spikes
  under skewed load or scan bursts) and full *mirroring* (high bandwidth and
  fast hits, but duplicates waste scarce fast-tier capacity). MOST mirrors only
  the small fraction of genuinely hot blocks into the fast tier while keeping
  their backing copy intact in the capacity/NVMe tier, avoiding demotion
  write-backs on fast-tier eviction and isolating one-hit scans from polluting
  the hot tier.
* **Clean-room license and dependency audit:** Upstream Cerberus / CacheLib C++
  artifacts were **not** inspected or imported. Everything in `pkg/most`,
  `most_test.go`, `tools/mostbench`, `docs/most.md`, and `docs/most.csv` is an
  independent, clean-room Apache-2.0 Go implementation written from first
  principles using only the Go standard library and existing `github.com/hanzoai/vfs`
  packages (`pkg/backend`, `internal/blake3`, `internal/storetest`).
  `go.mod`, `go.sum`, CI workflows, `tools/mutate/mutations.json`, and all
  frozen format-v2 wire layouts (`head.go`, `record.go`, `shard.go`, `pack.go`)
  are untouched.

## 2. HIP-1344 contract

`pkg/most` sits strictly below `vfs.Volume` as a `backend.Backend` host cache
tier (`vfs -> most.TenantBackend -> hanzoai/s3` inside `visor`):

1. **Unified single-engine implementation (`pkg/most/engine.go`, `pkg/most/policy.go`):**
   Both `PolicySelectiveMirror` (MOST) and the `PolicyTierMigration` baseline
   execute through the same `most.Engine`, wrapping one underlying
   `backend.Backend` and managing:
   * `ramPool`: byte-bounded host RAM pool with per-tenant LRU queues and
     deduplicated read-only sharing for `ScopeSharedImmutable`.
   * `nvmeTier`: disk-backed directory cache (`NVMeDir`) storing only sealed
     `ScopePrivate` frames or BLAKE3-verified `ScopeSharedImmutable` frames
     under content-hashed paths (`priv/<tenantHash>/...` and `shared/...`).
2. **Selective mirroring vs ordinary tier migration:**
   * **`PolicyTierMigration` (Baseline):** Exclusive tiering. Cold misses enter
     RAM directly; RAM eviction demotes victims to NVMe (`TierMigrationWrites`);
     subsequent NVMe hits migrate items back to RAM and delete the NVMe copy.
     One-hit sequential scans evict hot Zipfian items and trigger demotion writes.
   * **`PolicySelectiveMirror` (MOST):** Cold misses and shared toolchain reads
     default to `nvmeTier` only. A sliding-window / exponentially decayed
     hotness score (`HotThreshold`, default `2.0`, `DecayHalfLife = 30s`)
     gates admission to RAM. One-hit sequential scans never enter RAM; keys
     reaching `HotThreshold` are **mirrored** into `ramPool` while keeping their
     NVMe copy intact, so RAM eviction drops and zeros the RAM buffer with
     **zero demotion write-back** (`TierMigrationWrites == 0`).
3. **Fail-closed security & tenant isolation (`Cipher` + `BLAKE3` + `RevokeTenant`):**
   * `most.Cipher` matches `vfs.Cipher` structurally (`Seal`/`Open`) with zero
     circular import on package `vfs`, so `vfs.Hybrid(...)` plugs in directly.
   * Unregistered/revoked tenants or `ScopePrivate` requests without a cipher
     fail closed with `ErrTenantUnauthorized`.
   * Private payloads are sealed with an injective 32-byte length-framed BLAKE3
     AAD context `[tenantID | volID | key]` before `cipher.Seal(...)`; cross-tenant,
     cross-volume, or cross-key ciphertext swaps fail closed with
     `ErrIntegrityViolation`.
   * `ScopeSharedImmutable` verifies `SetSharedAccess` ACLs and pinned `BLAKE3`
     digests on every backend fetch, NVMe read, and RAM promotion (`ErrReadOnly`
     on tenant mutation attempts; `ErrIntegrityViolation` on digest mismatch).
   * `RevokeTenant(tenantID)` immediately zeros (`clear`) and purges all RAM
     buffers, removes all tenant private NVMe files from disk, and detaches
     shared references without disturbing healthy tenants.

## 3. Falsifiable hypotheses and matched baselines

* **Matched baselines:**
  1. `no_cache`: direct `backend.Backend` reads without host RAM or NVMe tiers.
  2. `tier_migration` (`PolicyTierMigration`): ordinary exclusive RAM/NVMe
     migration inside the same `Engine` (identical `Cipher`, quotas, locks, and
     disk-backed `NVMeDir`).
  3. `most_selective_mirror` (`PolicySelectiveMirror`): MOST selective mirroring
     with `HotThreshold = 2.0`.

* **Hypotheses:**
  * **H1 (Zero demotion write-amp):** Under mixed Zipfian + 1-hit scan traffic
    with RAM smaller than working set, `PolicySelectiveMirror` incurs
    `TierMigrationWrites == 0` while `PolicyTierMigration` incurs substantial
    NVMe demotion writes (`> 0`).
  * **H2 (Lower tail latency & higher RAM hit ratio under scan pollution):** By
    keeping 1-hit scans out of RAM and eliminating demotion writes from the
    read path, `PolicySelectiveMirror` reduces `P99` latency and increases
    `RAMHits` vs `PolicyTierMigration` on identical Zipfian workloads.
  * **H3 (Multi-tenant isolation & fail-closed security):** Under per-tenant
    RAM/NVMe quotas (`TenantStats`), a noisy neighbor flooding cold keys leaves
    Tenant A's `ResidentRAMBytes` and `RAMHits` intact; cross-tenant ciphertext
    swaps fail with `ErrIntegrityViolation`; `RevokeTenant` drops Tenant B
    residency to `0` bytes.
  * **H4 (Negative / boundary control — 100% zero-reuse cold scan):** When
    every key is accessed exactly once (`0%` reuse), MOST mirrors `0` keys
    into RAM (`RAMHits == 0`, `BackendGets == Ops`) and cannot reduce backend
    GETs. Against an unthrottled local store, NVMe admission overhead makes
    wall time slower than `no_cache` — while still avoiding RAM pollution
    (`0` KiB vs `192` KiB) and demotion writes (`0` vs `216`) vs
    `tier_migration`.

## 4. Empirical evaluation (real measurements)

All numbers below were measured directly on `linux/amd64` (2 CPUs) using real
disk-backed `NVMeDir` I/O, `time.Now()` wall-clock timings, real `storetest.S3`
HTTP counters, and `runtime.MemStats`. Workloads marked `sim2ms` explicitly
inject `2.0 ms` simulated S3 `GET` RTT via `most.ProbeBackend`; all other runs
have `0.0 ms` injected delay. Raw data is in [`docs/most.csv`](./most.csv).

### A. `tools/mostbench` comparative suite (`docs/most.csv`, `8 KiB` blocks)

| Workload | System | Delay (ms) | Ops | Wall (ms) | P50 (us) | P95 (us) | P99 (us) | Backend GETs | RAM Hits | NVMe Hits | Promotions | Demotion Writes | Res RAM (KiB) | Res NVMe (KiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zipf_shared_toolchain_unthrottled` | `no_cache` | 0.0 | 2400 | 66.22 | 22.9 | 698.2 | 1154.5 | 2400 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| `zipf_shared_toolchain_unthrottled` | `tier_migration` | 0.0 | 2400 | 1236.01 | 7.0 | 909.9 | 13978.0 | 421 | 1382 | 593 | 0 | 990 | 192.0 | 3136.0 |
| `zipf_shared_toolchain_unthrottled` | **`most_selective_mirror`** | 0.0 | 2400 | **87.56** | **5.5** | **456.8** | **2030.1** | **416** | **1492** | 489 | 489 | **0** | 192.0 | 3328.0 |
| `zipf_shared_toolchain_sim2ms` | `no_cache` | 2.0 | 480 | 293.70 | 2371.8 | 2956.5 | 4000.0 | 480 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| `zipf_shared_toolchain_sim2ms` | `tier_migration` | 2.0 | 480 | 169.64 | 271.5 | 3417.3 | 4051.0 | 190 | 201 | 85 | 0 | 251 | 192.0 | 1320.0 |
| `zipf_shared_toolchain_sim2ms` | **`most_selective_mirror`** | 2.0 | 480 | **149.51** | **65.9** | 3240.4 | 3953.0 | **189** | **229** | 55 | 55 | **0** | 192.0 | 1512.0 |
| `noisy_neighbor_tenant_a` | `tier_migration` | 0.0 | 120 | 54.74 | 5.3 | 62.3 | 124.3 | 0 | 120 | 0 | 0 | 144 | 128.0 | 0.0 |
| `noisy_neighbor_tenant_a` | **`most_selective_mirror`** | 0.0 | 120 | **44.10** | 8.5 | **16.8** | **90.3** | 0 | **120** | 0 | 0 | **0** | 128.0 | 128.0 |
| `negative_control_zero_reuse_scan` | `no_cache` | 0.0 | 240 | 6.68 | 22.9 | 29.2 | 64.7 | 240 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| `negative_control_zero_reuse_scan` | `tier_migration` | 0.0 | 240 | 59.05 | 226.4 | 445.4 | 678.3 | 240 | 0 | 0 | 0 | 216 | 192.0 | 1728.0 |
| `negative_control_zero_reuse_scan` | `most_selective_mirror` | 0.0 | 240 | 55.82 | 195.4 | 417.7 | 1108.0 | 240 | 0 | 0 | 0 | **0** | **0.0** | 1920.0 |

* **Memory & GC (`runtime.MemStats` on `zipf_shared_toolchain_unthrottled`):**
  `tier_migration` allocated `65,626,968 B` (`14` GCs) vs `42,335,728 B` (`9` GCs) for `most_selective_mirror` (`-35.5%` allocation churn due to zero demotion re-sealing).
* **Live `vfs.Volume` over `storetest.S3` (`most_test.go`):**
  * `TestMOSTSharedImmutableToolchainWithRealVolume`: 4 concurrent agents over
    2 cold-volume passes (`8` shared toolchain mounts + `4` private `vfs.Hybrid`
    volumes) cut live HTTP `S3.Requests()` from `16` (4 uncached mounts; `32`
    extrapolated for 8 uncached mounts) down to **`4`** (`RAMHits = 13`,
    `NVMeHits = 5`, `MirrorPromotions = 5`, `BackendGets = 4`,
    `TierMigrationWrites = 0`, `P99 = 4.04 ms`), with zero plaintext leakage on
    S3 objects or NVMe disk files.
  * `TestMOSTTenantIsolationPressureAndRevocation`: Tenant A keeps `32,768 B`
    resident RAM (`P99 = 702.19 us`) during Tenant B's 120-key flood;
    cross-tenant `vfs.Hybrid` ciphertext swap fails with `ErrIntegrityViolation`;
    `RevokeTenant("tenant-b")` zeros/purges Tenant B (`RAM = 0 B`, `NVMe = 0 B`).
  * `TestMOSTSelectiveMirrorVsTierMigrationZipf`: over disk-backed `NVMeDir`,
    MOST achieves `P99 = 1.215 ms` vs `4.235 ms` (`-71.3%`), `P95 = 345.2 us`
    vs `792.3 us`, `RAMHits = 1061` vs `1027`, and `0` vs `555` demotion writes.

## 5. Canonical v2 minimal integration path

`vfs` requires zero format-v2 wire or `Volume` core changes (`vfs -> most.TenantBackend -> hanzoai/s3 -> visor`):

```go
eng, _ := most.NewEngine(rootS3Backend, most.Config{
    Policy:         most.PolicySelectiveMirror,
    NVMeDir:        "/var/cache/visor/most-nvme",
    TotalRAMBytes:  512 << 20,
    TotalNVMeBytes: 8 << 30,
    HotThreshold:   2.0,
    SharedPrefix:   "toolchain",
})
_ = eng.RegisterTenant(agentID, volID, vfs.Hybrid(agentKey), 128<<20, 1<<30)

// Shared immutable toolchain mounted read-only across agents:
tcVol, _ := vfs.Open(ctx, vfs.Config{
    Backend: most.NewSharedBackend(eng, agentID),
    Cipher:  toolchainCipher,
})

// Agent private read-write workspace mounted over the same Engine:
wsVol, _ := vfs.Attach(ctx, vfs.Config{
    Backend: most.NewTenantBackend(eng, agentID),
    Cipher:  vfs.Hybrid(agentKey),
    Round:   1,
    Journal: agentJournalDir,
})

// On sandbox termination: immediately zero RAM and wipe tenant NVMe blocks.
_ = eng.RevokeTenant(agentID)
```

## 6. Reproducing

```sh
TMPDIR=/home/max/tmp go test -v -count=1 ./pkg/most
TMPDIR=/home/max/tmp go test -v -count=1 -run '^TestMOST' .
TMPDIR=/home/max/tmp go run ./tools/mostbench -csv docs/most.csv
```

# AKO: larger query chunks on both Sparks

## Outcome and decision

Both `192.168.2.178` and `.179` completed the same 31-workload attention
experiment. Summed equal-total-work 32K attention time decreases **5.48%**
and **5.29%**, respectively, with 8192 instead of 2048 query-token chunks.
This is not a faster attention implementation: the exact same c322 module
serves every chunk size.

The subsequent `.179` serving comparison uses one identical 8192-capacity
AOT bundle for both policies. Completion time decreases **24.10%** at
16384/64 C1 and **41.15%** at 32512/64 C1. All C1 output vectors match their
paired controls. At 4096/64 C1, completion time changes by **+0.14%**, not
a win. C1 decode TPOT is essentially unchanged; the gain is before the first
token.

The 32512/64 C2 timing decreases 40.56%, **but this cell is not admitted**:
16 request output vectors differ from the first control. Four differences
occur in the second **2048 control** run and twelve in the 8192 runs. All
differences are confined to C2. Production/default serving selection remains
unchanged; a concurrent/mixed execution numerical investigation is required.

Neither this comparison nor the microbenchmark is a fresh vLLM/SGLang result,
an independent model-quality validation, or a comparison with the previous
2048-capacity serving package. Do not add the earlier MLP fragment-window
gain to these percentages; its saved winning fold record was not explicitly
bound into this serving experiment.

## Fixed contract and parallel work

| Host | GPU UUID | Work this round |
| --- | --- | --- |
| .179 / spark-368c | GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6 | Attention vector, tail sanitizer gates, AOT preparation, unprofiled serving ABBA |
| .178 / spark-57f5 | GPU-9c3d3cf0-439a-5da2-67e9-20414255879f | Independent attention vector and two privileged counter captures |

Both are GB10/sm121, with CUDA 13.0.88. The nvcc executable SHA-256 is
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
The attention probe SHA-256 is
`014a0eeac3b27a3877eac139a0037e6615775dd0e75f477c41d442c7beccd66d`;
the reused c322 cubin SHA-256 is
`9aedf7709b833ad4ba05bf017d54199b33908e89d63c7fae51e73f94cf0a655d`.
The numerical law remains strict natural exponential with unchanged BF16
arithmetic. No relaxed arithmetic or model-name schedule branch was added.

GPU work is serialized **within each host**, not across hosts. Profiling on
.178 does not interfere with unprofiled timing on .179. The benchmark retains
a 32 GiB MemAvailable reserve, zero process swap, and externally bounded
user-systemd owners. The serving supervisor uses a 64 GiB ceiling and the
bridge 2 GiB; the parent orchestration uses 8 GiB. These are ceilings, not
measured GPU allocations. Minimum observed serving MemAvailable is
101,784,796 KiB (about 97.07 GiB), sampled every 500 ms.

Finite budget: 31 micro workloads per host, one terminal tail sanitizer set,
four fresh serving starts `[2048,8192,8192,2048]`, and two isolated counter
captures. No further schedule sweep is implied.

## Equal-total-work attention

Totals are 32768 and 32512 tokens. Chunks retain all previous history;
the 32512 tail replaces the final full chunk, not an extra synthetic chunk.
Per-head causal score counts are exactly **536,887,296** and **528,531,328**.
Chunk counts are 16/8/4 at capacities 2048/4096/8192. Tail lengths are
1792/3840/7936, ending at the same total sequence length.

Each chunk has five alternating repeat pairs, warmup and 30 timed repeats.
Both pair members use the same cubin. Sum the actual chunk timing vectors
before taking their median; do not multiply one convenient final-chunk
median by a chunk count. The two pair members are repeat controls, not
independent baseline/candidate implementations.

| Host | Total | 2K sum ms | 4K sum ms | 8K sum ms | 8K reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| .179 | 32768 | 68.922 | 67.000 | 65.276 | 5.29% |
| .179 | 32512 | 67.869 | 65.730 | 64.305 | 5.25% |
| .178 | 32768 | 66.233 | 64.306 | 62.606 | 5.48% |
| .178 | 32512 | 65.161 | 63.121 | 61.653 | 5.39% |

Do not infer speedup from cross-host absolute timing. Every host has its own
three-capacity control vector. All **62 workloads** passed full same-module
output equality and the sampled FP64 softmax oracle. That is not full
cross-chunk logit/token equality: each micro workload uses its own
deterministic current/history data.

The .179 Q7936/H24576 tail passed memcheck with leak checking, racecheck,
and synccheck: zero errors/hazards/leaks and empty captured stderr. .178 also
passed a tail synccheck repeat; it did not repeat all three sanitizer tools.
Minimum before/after micro MemAvailable was 121,782,596 KiB on .179 and
122,322,256 KiB on .178. These observations are not continuous GPU peak
memory measurements.

## Startup fixes needed to execute the serving experiment

The frozen source baseline is `9010b280`; captured overlays supply the
following fixes, without modifying frozen baselines or introducing hot-path
checks:

1. Fused export admits up to 8192 query tokens instead of 2048 and accepts
   the explicitly qualified CUDA 13.0.88 compiler alongside 13.1.115. Tests
   cover 8192 acceptance, 8193 rejection and unknown compiler versions.
2. The Qwen offline binder propagates `--device-target` through regeneration,
   compile receipts, tuning identity and bootstrap identity. It no longer
   silently assumes sm120 on a Spark sm121 export.
3. Binder errors report a bounded failure stage. This exposed `MemoryPlan`,
   rather than a generic publication error: an 8192-row vocabulary output
   alone exceeds the old 2 GiB activation limit.
4. `--activation-arena-gib 4` supplies the same immutable budget to planning,
   bootstrap-source identity and the release receipt. The planner reports
   **2,707,423,232 bytes** (2.52 GiB), below the explicit 4 GiB ceiling.
   The launch descriptor carries both the target and budget. Defaults remain
   the old target and 2 GiB budget.
5. The capacity-receipt compatibility helper accepts canonical BF16 CUDA
   targets instead of assuming sm120. Receipt digests and native preflight
   remain responsible for exact identity; accepting a parser value does not
   qualify hardware. Offline tests accept sm80/sm90/sm120/sm121 and reject
   sm75, leading-zero and malformed targets.

Serving manifest SHA-256:
`d66b674cef774ce083f4ce1bb35e268c83de51ef44a01e6a352f4034b1703436`.
The encoded fused runtime is **v11**, SHA-256
`dfb6e35a8940c4e060b8c6d7397f11bbb86360d03b4e6a78dcd72144e4973ac9`.
The argument changes do not relabel sm120 binaries or perform runtime JIT.

Owned source commits are `c2f8d890`, `0bac7b2c`, `0810a089`, and `412c3bdf`.
The unrelated dirty working tree was not uploaded or committed as a whole.
The MoonBit binder tests pass **8/8** locally and on Linux; the existing
capacity boundary and the additional target fixtures pass. Warning-denied
checks use the existing migration warning exclusions
`-79-29-25-20-92-14`; this is not a new full-repository release qualification.

## Same-bundle whole serving

Input vectors are `[4096]`, `[16384]`, `[32512]`, `[32512,32512]`; every
request emits 64 tokens with greedy sampling and EOS ignored. Each fresh
start has one warmup and three measured trials per cell. Each policy has
six measured trials per cell, with exact matching request bodies. Warmups
are excluded. All four supervisors acknowledged drain and closed the child
with exit code zero; runtime stderr is empty and the GPU returned idle.

| Input / concurrency | 2K / 8K wall ms | 2K / 8K output tok/s | Wall reduction | 2K / 8K TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| 4096 / C1 | 734.5 / 735.5 | 87.13 / 87.02 | −0.14% | 172 / 168.5 |
| 16384 / C1 | 2620 / 1988.5 | 24.43 / 32.19 | 24.10% | 1665 / 1021 |
| 32512 / C1 | 7619 / 4484 | 8.40 / 14.27 | 41.15% | 6147 / 3008.5 |
| 32512 / C2, **diagnostic only** | 15662 / 9309.5 | 8.17 / 13.75 | 40.56% | 9334.5 / 4645.5 |

C1 median TPOT is 8.619/8.627 ms, 14.897/14.984 ms, and
23.048/23.056 ms. The improvement does not accelerate long-context decode.
At C2, the early request's TPOT includes waiting while the other request
prefills; a per-request median here is not isolated decode-kernel duration.

All **36 measured C1 request vectors** match the corresponding first control.
For C2, 16 of 24 measured request vectors differ: four in the second control,
twelve in the 8K arms. First differences occur at tokens 2–6; affected
vectors differ in 1 or 39–41 of their 64 tokens. Thus this is not just a
reported throughput rounding difference. The audit preserves exact request
bodies, token identities, first-difference indices and token timestamps.
It does not establish whether this is a runtime ownership bug or permitted
numerical variation under mixed execution; neither is assumed away.

The client uses synthetic varied token arrays from a small vocabulary
alphabet, not a natural-language quality corpus or a broad model suite.
Longer input vectors improve length coverage, not workload diversity.

## Counter capture scope and remaining attribution

.178 profiled the first selected c322 kernel at matched retained history
24576, once for Q2048 and once for Q8192. These are **not equal-work** calls
and are not used as unprofiled timing results.

| Metric | Q2048/H24576 | Q8192/H24576 |
| --- | ---: | ---: |
| Selected symbol | lunaflux_attention_prefill_tile_compiler_v1 | same |
| Grid / block | (63,16,1) / (128,1,1) | (159,16,1) / (128,1,1) |
| Warp instructions | 986,105,728 | 4,381,476,736 |
| Registers/thread | 235 | 235 |
| Tensor-active | 57.79% | 57.69% |
| Active-warps occupancy | 15.97% | 16.50% |
| Profiled duration | 7.41 ms | 32.35 ms |
| Requested DRAM/long-scoreboard/barrier metrics | **n/a** | **n/a** |

Both probes still pass their same-module comparison and sampled oracle.
Missing metrics are unavailable, not zero. The capture does not prove
reduced memory dependency or barrier stalls, nor explain the whole serving
gain. Similar per-kernel tensor activity is consistent with chunking changing
the executed work chain rather than a radically improved latency-hiding
pipeline. Fresh whole-chain/step attribution is still needed.

## Preserved records

- .179 experiment: `/home/wlc004s/lunaflux-ako-query-chunks-20261005.PCwRF4kW`.
- .178 repeat: `/home/wlc003s/lunaflux-ako-query-chunks-20261005.thHSoVZ6`.
- .178 counters: `/home/wlc003s/lunaflux-ako-query-chunks-profile-20261005.VqDeMdwE`.
- Reports: `equal-work.json`, `e2e-summary-audited.json`, all raw request
  bodies/SSE/token vectors, failed attempt outputs, compile recipes and
  source-overlay hashes. Failed pre-budget/pre-target runs are not relabeled.
- Secondary measurement archive SHA-256:
  `993f7686a7ad434222e5c1907494d0ec10e7f2c269dc3aabddd8d81b46c20d67`.
- Primary measurement archive SHA-256:
  `b6e927c70a4654073e520745a6d6bb1ef6a73828c5ef8f22bcb88a1c48b27ca9`.
- Counter report SHA-256, Q2048:
  `1273c279297a7bb29533cbe87633e1e1ab58d4d2b29d4eb3f55c0da8bd2b93fe`;
  Q8192: `fd01c20659547bf036b523662e4fa76136362a381738dc98bdd90fe1d96f6e10`.
- Local verified downloads: `/tmp/lunaflux-ako-dual-results-20261005.2nlGEAgM`.

The primary measurement archive excludes large weight files and serving
executables, whose exact hashes remain in `REMOTE_ONLY_ARTIFACTS.sha256` and
whose bytes remain remote. It retains source, receipts, cubins and measured
outputs. This is an offline AKO result, not a promoted deployment.

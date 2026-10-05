# Dual-Spark epoch-local joint consumer, 2026-10-05

## Outcome

Keep c30322. Final c130322 was slower in all six unprofiled cells; none passed
the fixed 3% paired improvement threshold. Both Sparks ran concurrently, then
`.178` ran sanitizer checks while `.179` captured matched counters. This is
selected attention-kernel work, **not** new serving throughput or a refreshed
vLLM/SGLang comparison. No production route changed.

## Hypothesis and budget

Cross-epoch query retention reduced shared loads but increased registers and
supporting instructions without a robust timing benefit in the previous
[query-lifetime experiment](BENCHMARK_AKO_QUERY_LIFETIME_2026-10-05.md).
Test joint right-column consumption with **zero cross-epoch retention** instead:
load one query fragment inside each KV epoch, consume its right columns, then
end its lexical lifetime before loading the next reduction fragment.

The immutable lifetime IR separates retention from consumer fanout; CUDA
lowering realizes the epoch-local scope. Arithmetic fold order, numerical law,
transfer barriers, launch geometry and production selection stay unchanged.
The new general candidate is strict 100322 / approximate 130322, not a Qwen rule.

Budget: one alternative, three Q2048 workloads (R1/H28672, R2/H28672,
R2/H8192), five alternating pairs per workload on each Spark. Compile once on
`.179`, run both hosts' timing concurrently, then split sanitizer checks on
`.178` and matched instruction counters on `.179`. GPU jobs have 16 GiB memory
caps, zero swap and a 32 GiB MemAvailable reserve. No compilation during timing.
Only a robust win can justify a subsequent serving-propagation test.

## Preparation fixes

The first preparation
attempt failed before GPU work because macOS AppleDouble source sidecars were
included in the transfer and the helper build exhausted its process limit.
Preserve that attempt. The retry uses a fresh source copy with sidecars excluded
and explicitly limits MoonBit build jobs, without changing the GPU budget.

The exporter initially rejected the approximate epoch-local identity: candidate
enumeration had been extended but numerical-policy admission had not. The same
pure catalog now admits the new IDs only with explicit approximate permission;
CLI regression tests cover refusal without permission and successful opt-in.

Review also found that the initial `50000 + base_id` allocation could collide
with existing approximate retained IDs for other tile shapes (for example,
82000). Use the disjoint `100000 + base_id` range, preserving every existing
ID. A uniqueness regression now checks the entire generated strict and
approximate query-owned frontier across query buckets. The initial measured
diagnostic 80322 is retained separately; final identity 130322 is re-exported
before comparing compiled artifacts. This is startup/compiler work, not a
token-step identity check.

## Final unprofiled measurements

The final-ID repeat is an identity correction, not another schedule search.
Its generated CUDA source is byte-identical to the initial diagnostic. Both
final AOT compilations produce the same cubin. The different cubin hash from
the earlier diagnostic warrants fresh timing/correctness/profile checks; all
were repeated against the final artifact, with the old results retained.

Times below are median microseconds. Paired change is the median of the five
`candidate/baseline - 1` observations, not a ratio of the two table medians.

| Host | Requests / history | Baseline µs | Epoch-joint µs | Paired time change |
| --- | --- | ---: | ---: | ---: |
| .178 | 1 / 28672 | 7123.23 | 7340.93 | +3.20% |
| .178 | 2 / 28672 | 7256.41 | 7399.63 | +2.55% |
| .178 | 2 / 8192 | 2099.99 | 2115.16 | +0.61% |
| .179 | 1 / 28672 | 7469.30 | 7663.98 | +2.48% |
| .179 | 2 / 28672 | 7342.63 | 7564.60 | +2.52% |
| .179 | 2 / 8192 | 2171.37 | 2213.09 | +1.95% |

The initial-ID diagnostic also lost all six cells (+0.82–4.76%). It is not
pooled into these final-artifact measurements or discarded as noise.

## Executable interpretation

Matched Q2048/R2/H28672 captures reproduce these exact counts for both exports:

| Metric | Baseline c30322 | Epoch-joint |
| --- | ---: | ---: |
| Registers/thread | 234 | 246 |
| Dynamic shared bytes/CTA | 49,168 | 49,168 |
| Resident CTAs/SM | 2 | 2 |
| Executed warp instructions | 936,583,040 | 913,620,864 |
| Non-transposed LDSM | 37,396,480 | 37,396,480 |
| Transposed LDSM | 29,917,184 | 29,917,184 |
| HMMA | 119,668,736 | 119,668,736 |
| Async LDGSTS | 14,958,592 | 14,958,592 |
| Workgroup barriers | 2,808,832 | 2,808,832 |
| LOP3 | 23,710,720 | 7,665,664 |
| IMAD | 7,553,024 | 106,496 |
| IADD3 | 10,828,544 | 9,620,224 |
| MOV | 100,893,696 | 99,008,512 |
| NOP | 25,242,624 | 34,591,744 |

Total instructions fall 2.45%, chiefly from addressing. **Shared matrix-load
counts do not fall**: baseline lowering already permits backend reuse of the
query operand. Shortening its source scope and grouping the RHS consumers does
not eliminate additional physical loads. The new instruction region instead
uses 12 more registers and 9.35M more executed NOPs. It still has two resident
CTAs and zero spills/local bytes, so an occupancy collapse cannot explain it.

Do not turn those NOP counts into stall cycles or assert that barrier delay
caused the entire loss. The two counter captures' latency/stall shares vary:
initial average warp latency 6.49→7.50; final 7.09→6.69. Final instrumented
duration even favors the candidate (8.70→8.30 ms), unlike both unprofiled
five-pair campaigns. Issue-active percentage falls in both captures
(29.42→28.15% and 29.45→25.76%). These observations motivate inspecting
instruction-region dependency/scheduling and register realization, not another
retention-size sweep or a claim that one profiler percentage explains timing.

The immutable semantic fold and physical lifetime plan stay distinct from
CUDA instruction lowering. No new IR layer, model-specific choice, runtime
JIT or token-step validation was added. The new alternative remains diagnostic;
default candidate selection and all pre-existing identities are unchanged.

## Correctness, limits and retained results

All final 30 timing pairs match the frozen approximate-law baseline bitwise;
sampled FP64 oracle error is at most 0.000377474, below the fixed 0.003 ceiling.
Q129/R2/H128 memcheck, racecheck and synccheck each pass with zero errors or
hazards. This does not establish strict-law equivalence or whole-model quality.
Both GPUs were idle at completion. Minimum recorded MemAvailable was
122,332,472 KiB on `.178` and 122,042,960 KiB on `.179`, above the 32 GiB reserve.

174 affected native tests pass with existing warnings 20/79/29/25 excluded;
scoped warning-denied check uses existing dependency exclusions 20/79/29.
Both `.mbtx` helpers pass native warning-denied regression tests without
exclusions, and affected files pass format checking. Package info introduces
no public API signature changes. The full dirty repository is not claimed
warning-clean and unrelated changes are untouched.

Final local root: `/tmp/lunaflux-ako-epoch-joint-20261005.tVUzPfwp/final`.
Remote roots:

- `.178`: `/home/wlc003s/lunaflux-ako-epoch-joint-20261005.6Fm6crp4/experiment-final`.
- `.179`: `/home/wlc004s/lunaflux-ako-epoch-joint-20261005.xcZDqwJx/experiment-final`.

Both downloaded archive hashes match, and all 55 `.178` and 1,099 `.179`
manifest entries verify locally:

- `.178` archive: `b5b16da10cc64e0b9e0399baa7a422e8a47a06d261e2425ef564b7932579c73d`.
- `.179` archive: `b2e875cf772627641d45bafb5012622e855183e9a959062791ac19ca4d5ac66d`.
- Baseline cubin: `66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
- Final epoch-joint cubin: `a01c95ad28544f2302da0f0e7402356a126f49d31e70417a3131dfc3340c7f72`.
- Epoch-joint source: `810cfc10db49750dc9011193dc385c58c0b432b8cd9e9be897e5fd5492517d88`.

Preserved initial-ID measurements are in the parent local directory and the
remote `experiment` (`.178`) / `experiment-v2` (`.179`) roots; both original
archives and manifests also verify. Their archive hashes are
`2e6b6ff16226cd8ac7cdab6a222043470c79a6d24b38e289c4159b84010c041b`
and `c6ad265d5be4fd4733cd3d3a05b7dde79debecafc39053c1375ed479318e3fd3`.
The preserved pre-GPU failure/input archive verifies as
`716db7e3ce8f43f5cd5ad5b0936ca6ef165aa7b06676ec03f68c315c24fdcc26`.

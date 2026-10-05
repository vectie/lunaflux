# Dual-Spark prefill route isolation

Both Sparks ran independent experiments concurrently. This continues
`BENCHMARK_AKO_CHUNK_LOGITS_2026-10-05.md`, reuses its frozen Qwen3-0.6B
BF16 model/AOT bundle, and changes no production default or numerical tolerance.

## Concrete finding

The two chunk policies did not select the same prefill chain. The old 2K
final prefill used executor owner 63, the two-part split-prefill chain. The
8K final prefill used owner 33, ordinary unsplit c322. On the same model,
disabling both prefill split branches in a disposable worker eliminates the
previous cross-chunk token and tracked-logit differences on both saved prompts.

This is a selection/numerical-law problem, not a sampler mismatch. It also
explains why checking only c322 chunk invariance was insufficient. The previous
claim that the large chunk gain could be explained by chunking alone is not
established: chunk size changed the selected attention chain too.

## .179: exact-model route intervention

Root: `/home/wlc004s/lunaflux-ako-chunk-trace-20261005.gcDzIyxE`.
GPU: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6` (GB10, sm121).
Frozen input/runtime root:
`/home/wlc004s/lunaflux-ako-query-chunks-20261005.PCwRF4kW`.
Bundle SHA-256:
`dfb6e35a8940c4e060b8c6d7397f11bbb86360d03b4e6a78dcd72144e4973ac9`.

The source copy gains route tags and the existing synchronous logit readbacks.
The only route intervention is disabling ordinary/deep prefill partition
selection. Decode, model weights, kernels, graph capture, chunk budgets and
accuracy tolerances are unchanged. The optional wide-query owner is absent
in this bundle; the experiment uses the ordinary unsplit owner, not a new
wide-query artifact. The initially named `--wide-only` diagnostic option has
therefore been renamed `--unsplit-prefill` in the committed runner.

Route tag 15 identifies the split slot owner as 63, deep owner 72, no wide
slot owner, minimum split context 256 and partition count 2. At the final
2K bucket (246), the intervention selects unsplit owner 27 instead of 63;
the 8K bucket (288) remains owner 33. These indices refer to this exact build.

Two saved 32,512-input/64-output prompts are run sequentially, twice per
chunk arm: eight 64-token vectors, 512 outputs total.

| Check | Result |
| --- | --- |
| Server tokens versus client tokens | Exact agreement for all eight vectors |
| GPU versus CPU argmax | 0/512 disagreements |
| Within-arm repeat token/6-tracked-logit differences | 0 |
| Cross-chunk token differences, both prompts/repeats | 0 |
| Cross-chunk tracked BF16 logit bit differences | 0 |

Previously prompt 1 differed in 44/64 tokens starting at sample 2, and
tracked-logit differences began at sample 0 for both prompts. The intervention
removes those differences on this finite fixture. It does not prove full-logit
equality, arbitrary-input accuracy, concurrency parity or vLLM/SGLang parity.
The synchronous readbacks make these runs unsuitable for timing claims.

Build, preflight, capacity regeneration and replay finish successfully.
The preparation unit uses 12 GiB/no swap/600 seconds; serving uses
48 GiB/no swap/600 seconds. MemAvailable before serving is 122,822,032 KiB.

## .178: identical-input selected-chain comparison

Root: `/home/wlc003s/lunaflux-ako-prefill-split-20261005.J4h3xKQB`.
Authoritative trial: `runtime-geometry/trial`.
GPU: `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f` (GB10, sm121).
CUDA 13.0.88 compiler SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

The probe compares ordinary c322 (query/KV tiles 64/64) against the admitted
two-part prefill chain (64/128), including its partial and merge launches and
136,314,880-byte workspace. Partial consumes tile metadata; merge consumes
separate CSR row offsets, exactly as the runtime does. Immutable synthetic
queries/KV, positions and nontrivial physical-page mapping are shared.

The first trial overlaunched split X using metadata capacity. It is preserved,
not authoritative. The corrected probe mirrors the runtime's
`QueryTileCappedGridX` for both split launches, versus
`QueryMetadataCappedGridX` for c322. For the final 2K bucket the grids are
63×16×1 (ordinary), 32×16×2 (partial), 32×16×1 (merge).

Five alternating paired trials per workload, median complete-chain GPU time:

| Actual queries / prior history | Unsplit c322 | Split-2 chain | Split/unsplit |
| --- | ---: | ---: | ---: |
| 1,792 / 30,720 | 7.002 ms | 21.000 ms | 3.00× |
| 2,048 / 28,672 | 7.620 ms | 20.811 ms | 2.73× |
| 7,936 / 24,576 | 26.415 ms | 65.961 ms | 2.50× |

All pairs pass the unchanged 0.003 differential/sampled FP64-oracle bounds,
with unchanged KV and empty stderr. The first shape has maximum pairwise
error 0.000488281 and sampled oracle error 0.000342709; it is not bitwise
equal. The split partial reports 255 registers, one resident block/SM and
zero local bytes. Those resource observations do not, by themselves,
attribute the entire timing gap; no new Nsight stall-counter capture is made.

The runner requires a 32 GiB MemAvailable reserve before/after every workload;
observed available memory remains above 123,048,844 KiB. External unit limits
are 8 GiB/no swap/600 seconds. Native MoonBit automation is built on .179 and
transferred to .178, which has no Moon installation.

## Decision and remaining work

Reject split-2 for these populated query frontiers. It is slower and changes
the model's BF16 trajectory when selected across chunk policies. Preserve
partitioned attention for under-populated frontiers rather than deleting its
capability or globally forcing one kernel.

The pure fallback currently decides from context/query bucket bounds and a
derived context threshold; it does not account for these measured chain costs.
The next production change should feed the existing startup route table with
whole-chain workload/resource/numerical results, retaining the compiler's
semantic → legal schedules → measured selection → effects → lowering layers.
Do not add model-name cases, request-path profiling or tolerance relaxation.
An uninstrumented paired serving replay and C2/mixed-input parity are still
required before claiming a whole-engine speedup or changing a runtime default.

Automation additions: diagnostic route tagging/intervention, compact audit
summary, prefill-partitioned support in the existing selected-policy probe,
and nested probe retention in the archive helper. Focused warning-denied
native script checks/tests pass; no public production API changes.

.179 archive SHA-256:
`86b20efa20c431b7ae36196345f92554baec2c150d961ff19fbb2ef6e41dd5d0`.
Downloaded without overwriting to
`/tmp/lunaflux-prefill-route-transfer-20261005.pAAPJ9ZC/unsplit-serving.tar.gz`;
the local hash matches. The .178 provisional and corrected outputs are retained
together; the corrected trial is explicitly authoritative. Its corrected
executable is `runtime-geometry/probe`; immutable kernel/spec arguments still
point to the original experiment root (identical copied artifacts also remain
under `runtime-geometry`). Do not interpret the path prefix as old probe code.

.178 archive SHA-256:
`8519b4c3dfa3d311a3c8e3eb45ee2d7f7f8bb33cd3d4004e28a3bbb5003bb28a`.
Downloaded to the same local transfer directory as `split-attention.tar.gz`,
with a matching local hash. All 171 serving and 83 attention archive manifest
entries verify after extraction into separate directories. Both GPUs are idle
after the experiments.

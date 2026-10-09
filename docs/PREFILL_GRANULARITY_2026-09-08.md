# Prefill work granularity experiment

## Design

The next long-prefill experiment changes work partitioning, not attention
arithmetic. `scheduler/work_plan.prefill_query_count` is already a pure function
of progress, remaining budget, chunk bound, and KV capacity. There is no
256-token constant in that function. The previous release's AOT/profile and
workspace envelope supplied that bound.

For one request, coalescing adjacent prefill chunks must preserve the ordered
sequence of token positions and KV writes. Only the final chunk observes a
logits row. This is an effect-preserving partition transformation, not
floating-point reassociation or a new CUDA-specific scheduler heuristic.
Different query shapes may nevertheless select different numerical kernels;
whole-model token agreement must be measured rather than inferred from the
range law.

Build a separate 1,024-query-token release with matching AOT artifacts,
activation/workspace, descriptor, and worker capacity. Hold that release fixed
while measuring `(step budget, per-request chunk)` values `(256,256)`,
`(512,512)`, `(1024,1024)`, and `(1024,256)`. The last arm distinguishes
coalescing one request from packing more requests into a step. Preserve
decode-first reservation, aging, cancellation, and page ownership. Never
override scheduler limits beyond the admitted device profile.

All runs use Qwen3-0.6B BF16 on the isolated RTX 5060 Ti, the compiler kernels
from `67f07be`, the same partial QKV ingress route, fixed input IDs and output
counts, prefix reuse disabled, and C1/C8. Production is not modified. This is
an experiment until correctness and the workload-vector measurements complete;
larger is not assumed better.

## Executor bug found by the experiment

The first 1,024-token release reached Ready, then failed on its first 59-token
request. Isolated per-kernel diagnostics identified operation 5, CUDA error
700. Function inspection showed `lunaflux_attention_prefill_tile_compiler_v1_c314`
(64 query rows) being launched with the baseline 32-row rule: grid X=2 and
45,056 shared bytes. Its own rule requires grid X=1 and 73,728 shared bytes.

Both the generated primary source and its exact packaged CUBIN passed a
standalone exhaustive numerical check and compute-sanitizer memcheck for this
shape. Those checks initially exercised the primary kernel; querying the
executor's actual function exposed the wide-variant mismatch. Ready alone did
not exercise this execution path.

Commit `510d07b` gives wide-prefill graphs a separately derived launch-rule
value. A phase enum selects baseline, decode, wide-prefill, or split-prefill
rules. Missing variant rules preserve the selected step's exact geometry;
they never borrow another kernel's rule. The rule carries tile width, block
geometry, and shared memory together. Both BF16 preparation routes supply the
wide steps. This is startup-only composition; no new token-step allocation,
filesystem operation, diagnostic synchronization, or CUDA arithmetic change.

Regression coverage includes 59-token and 1,024-token bucket bounds, baseline
immutability, and absent variant metadata. All 162 device-step tests pass;
the 5 work-plan and 132 scheduler tests also pass. Temporary error logging and
per-kernel synchronization exist only in separate diagnostic sources, not in
this commit. Corrected end-to-end measurements are pending below.

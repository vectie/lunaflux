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

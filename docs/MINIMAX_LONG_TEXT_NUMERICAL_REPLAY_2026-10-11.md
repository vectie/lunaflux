# MiniMax longer-text numerical localization — 2026-10-11

The 140-token original-tokenizer input completes the native fifty-layer text
encoder and an independent original-weight ATen encoder on Spark `.179`.
This input is mostly repeated `Hello ` followed by the corpus request: it tests
a wider shape, not a diverse caption distribution. No production arithmetic,
CUDA lowering or runtime hashing is added by these diagnostics.

## Whole-prefix differences

Every hidden state contains 140 × 5,120 BF16 values. Native embedding and the
independent embedding match at all 716,800 values. First divergence is layer 1.

| Frontier | Global relative L2 | Row 5 relative L2 |
| --- | --- | --- |
| Layer 1 | 0.1854% | 0.1846% |
| Layer 6 | 0.5096% | 0.4380% |
| Layer 7 | 0.8362% | 11.1462% |
| Layer 50 | 0.9633% | 22.1058% |

Layer 50 has RMSE 0.9991901 and maximum absolute difference 224. Row 139's
relative L2 is 1.2314%; row 5's cosine is 0.9763587. Large channels dominate
the global norm, so global relative L2 alone conceals substantial row errors.
These are measured differences, not an acceptance tolerance or parity claim.

Turning off reduced-precision BF16 projection reductions in the reference
produces the **same final 716,800 BF16 values** for this shape. That setting
does not explain this case's full-prefix discrepancy; the earlier three-token
sample's 11% RMSE reduction must not be generalized to longer inputs.

## Same-input layer 7 replay

`replay-layer` evaluates one complete independent layer, including its own
normalization, Q/K/V, rotary, attention, output, residual and MLP operations.
It consumes an explicitly named prior hidden state, not a newly propagated
reference prefix. The native hidden-6 input and the captured replay input match
bitwise at every value. A second replay on reference hidden-6 reproduces the
ordinary reference hidden-7 bitwise at all 716,800 values.

| Comparison | Row 5 relative L2 | Meaning |
| --- | --- | --- |
| Native layer 7 vs reference layer 7 on native hidden-6 | 0.087264% | Local implementation difference with identical input |
| Reference layer 7 on native hidden-6 vs reference layer 7 on reference hidden-6 | 11.136766% | Same implementation, different propagated input |
| Native hidden-7 vs propagated reference hidden-7 | 11.146231% | Original whole-prefix comparison |

These relative norms are not additive. The replay shows that the large jump
can occur in the reference itself when the prior input changes. It does not
prove every native operator correct or eliminate the upstream numerical/quality
issue, but rules out attributing this jump solely to a layer-7 indexing bug.

## Where input differences amplify

Compare all sixteen snapshots from the two independent reference layer replays.
Only their input hidden states differ; weights and arithmetic stay identical.

| Operator frontier | Row 5 relative L2 |
| --- | --- |
| Input hidden-6 | 0.4380% |
| Attention output | 0.8658% |
| Attention residual | 0.4610% |
| MLP input normalization | 0.7647% |
| Gate/up/SiLU activation product | 13.5684% |
| MLP down | 15.8691% |
| MLP residual | 11.1368% |

For this row, the main amplification is in the MLP chain, not attention.
The activation product combines two projections and SiLU; it does not yet
isolate the individual gate/up projection or multiplication contribution.
The additional precision probe below separates those projections. Diverse
caption/media-quality tests remain necessary before defining acceptance.

## Projection precision versus input sensitivity

Two new independent probes consume the respective captured layer-7 operator
inputs. Each executes Q/K/V, output, gate, up and down with library BF16,
full-reduction BF16, F32 and F64 projection arithmetic. Original layer-7 weights
are used; all four outputs are rounded to the **same BF16 output boundary**.
The probes do not form a full F64 encoder or an unrounded F64 MLP chain.

| Row-5 comparison | Gate relative L2 | Up relative L2 | Down relative L2 |
| --- | --- | --- | --- |
| BF16 vs F64, same native-prefix operator input | 0.01919% | 0.01980% | 0.003827% |
| BF16 vs F64, same reference-prefix operator input | 0.01657% | 0.009697% | 0.004149% |
| Different operator inputs, same BF16 projection | 0.35780% | 0.53283% | 15.86910% |
| Different operator inputs, same F64 projection | 0.35764% | 0.53277% | 15.86910% |

The down input is the already captured activation product for each replay.
Its nearly unchanged sensitivity with wider projection arithmetic rules out
down's projection precision as the explanation for this row's large jump.
Gate/up projections themselves show much smaller differences than their
13.5684% activation-product difference. That localizes the next investigation
to how changed operands combine through SiLU/product and propagate from earlier
layers, rather than changing every projection's accumulation policy blindly.
It does not establish full-model quality or prove all native operators correct.

Both precision probes exit 0, are not OOM-killed, report zero retained tensors,
and leave no GPU owner. Each retains at most the enforced 2-GiB tensor budget
inside an 8-GiB/no-swap container; actual container peak was not captured.
The twenty-four MLP comparisons are CPU-only. Their comparator also fixes an
unrelated diagnostic defect: nonzero error against an all-zero reference now
reports infinite relative error rather than zero. Existing 140-token norms are
nonzero, so this fix does not change the previously recorded measurements.

## Execution and reproduction

The native run uses the existing `e704acf5` ARM binary, an exact 140-row text
AOT compiled for `sm_121`, a 60-GiB device budget and an 80-GiB/no-swap user
unit. It reports 51,506,191,840 weight bytes and 16,702,016 request bytes,
finite final output and successful exit. Unit runtime is 71.128 seconds;
reported cgroup host-memory peak is 2.8 GiB and swap peak is zero. That host
peak does **not** measure all CUDA unified-memory residency.

The independent encoder and layer replays use an 8-GiB/no-swap container and
2-GiB retained-tensor budget, one GPU workload at a time. Both layer replays
exit 0, report retained bytes zero, are not OOM-killed, and leave no GPU owner.
Reference container peak memory was not recorded. Their lifetimes are not a
matched native/reference throughput benchmark.

Raw native layers: `/tmp/lunaflux-minimax-text-encoder-long140-20261011-v1-179/layers`.
Full references: `/tmp/lunaflux-minimax-aten-reference-long140-20261011-v1`.
Whole-prefix reports: `/tmp/lunaflux-minimax-text-long140-comparison-20261011-v1`.
Layer replays and identity comparisons:
`/tmp/lunaflux-minimax-aten-reference-layer7-20261011-v1`.
Operator sensitivity reports:
`/tmp/lunaflux-minimax-layer7-input-sensitivity-20261011-v1`.
Layer-7 seven-projection precision probes:
`/tmp/lunaflux-minimax-aten-reference-layer7-precision-20261011-v1`.
Separated precision/input-sensitivity comparison reports:
`/tmp/lunaflux-minimax-layer7-projection-comparison-20261011-v1`.

The driver lives in the separate `benchmarks/checkpoint_aten_reference` module.
`scripts/replay-minimax-encoder-layer.mbtx` collects one-layer outputs;
`scripts/compare-minimax-encoder-operators.mbtx` accepts an explicit comparison
scope and `input-hidden` frontier. Variable-row native collection requires exact
matching `SOURCES.v1` geometry and computes output extents before execution;
it does no artifact checksum scan. These tools are not startup prerequisites.

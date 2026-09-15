# Actual-model QKV activation diagnosis

This is numerical diagnosis, not a throughput benchmark or completion of the
compiler refactor. It uses real Qwen3-0.6B weights and the established partial
fusion runtime, with synthetic token-ID requests at 3072 input / 32 output,
C1 then C8, one trial each on the RTX 5060 Ti.

The isolated worker executes the selected launches eagerly. Immediately before
operation 2 (first-layer QKV) it synchronizes and copies the last query token's
input row; immediately afterwards it copies that row's output. Offsets and row
strides come from the memory plan. Reading before and after the operation is
essential because later operations reuse activation storage. The diagnostic is
not installed in the production tree and its timings are not serving results.

## Observation

- 77 input/output pairs: input width 1024, output width 4096 (BF16).
- 77 pairwise comparisons have exactly equal captured input bytes.
- 29 of those comparisons have different output bytes, affecting at most five
  output components per pair.
- Every unequal comparison crosses single-token versus multi-token execution.
  Captured batch token counts include 1, 2–8, 37, 1024 and 2048.
- For example, epochs 3 and 66 have identical inputs, batch token counts 1 and
  8, and four different QKV components: 434, 462, 1065 and 3561.

Thus this is no longer merely a synthetic-vector hypothesis: actual model
activations show schedule-dependent QKV results despite equal inputs. Together
with the existing single-row/tree versus matrix reduction implementation, this
supports a reduction-order investigation. It does **not** prove that QKV alone
causes the previously observed final-token flip, nor establish an error bound
against an independent high-precision dot-product oracle. Near-zero results
must not be judged only by BF16 ULP distance.

### FP64 reference follow-up

`check_activation_dot_oracle.mbtx` now reads the actual BF16 Q/K/V rows from
the numeric safetensors file and computes sequential FP64 dots with captured
inputs. The tensor names are explicit inputs; row selection respects the
concatenated matrix dimensions and file offsets. It does not depend on the
CUDA generator's indexing or accumulator implementation.

Across the 82 changed-component observations in those 29 pairs, the
single-token result is closer to the reference every time. These observations
include repeated inputs; they are not 82 independent statistical samples.
The maximum observed absolute error across both paths is
0.000976871990133077. The maximum error divided by sum of absolute products is
0.00027175375139377284, including BF16 output rounding.

For component 4080 at epochs 18/38, the reference is
`-1.7276033759117126e-7`; single-token gives `-1.7136335372924805e-7`
(absolute error `1.3969838619232178e-9`) and matrix gives
`-2.5890767574310303e-7` (absolute error `8.614733815193176e-8`).
The sum of absolute products is about 3.086: cancellation explains why the
relative-to-result/ULP difference looks much larger than its absolute error.

This does not establish a universal numerical bound or end-to-end token
equivalence. Forcing the matrix path solely to match batch outputs would
discard the more accurate result in these observations. Final-logit propagation
and numerical acceptance remain open; no production policy was changed.

Oracle result: `/tmp/lunaflux-activation-campaign-20260915-r2/dot-oracle.json`,
downloaded as `/tmp/lunaflux-dot-oracle-20260915-r2.json`, matching SHA-256
`8ba8ea53194d630b7c971ba827a19d476105ba188b151a33edc2e30158774a01`.
The oracle regression checks two concatenated matrices, signed BF16 operands,
exact known dots, error ranking, and rejection of incomplete weight coverage.

The parser compares only complete single-input/single-output captures and
rejects missing, reordered or truncated words. It is not a general fused-span
output mapper. Full ingress fusion must not reuse operation-2 output assumptions.

## Reproduction and validation

### Final-normalization capture attempt

An isolated clone of the same historical diagnostic source was rebuilt with
the capture operation changed from 2 to 281. The C1/C8 requests completed and
the runner stopped its processes, but no activation events were emitted.
Consequently this attempt supplies no final-normalization numerical result.
The reusable runtime contains residual/RMSNorm fusion; the standalone
operation-ID assumption is not a valid way to observe an interior fused
operation. Follow-up must capture an actual launch boundary with its correct
output mapping (or the head input alone), not interpret absent events as
equality. Run: `/tmp/lunaflux-finalnorm.LBtNiA`.

The capture parser now also requires input/output position and batch context
to match and only compares equal activation identities. Its regression rejects
an otherwise complete pair whose output position changed. This is offline
diagnostic validation, with no added token-step work.

Tools: `benchmarks/gpu_pipeline/install_activation_readback.mbtx`,
`summarize_activation_readback.mbtx`, `test_activation_readback.mbtx`.
The installer requires an isolated execution-trace tree and cannot be combined
with the fixed-graph repetition diagnostic. Both ordinary and numeric-BF16
preparation retain the diagnostic memory plan; the first attempt missed the
BF16 path and failed before capture. That failed run remains separate.

Successful run: `/tmp/lunaflux-activation-campaign-20260915-r2`.
Worker source: `/tmp/lunaflux-activation-readback-v2-source-20260915`, derived
from the historical margin source, **not** a fresh current-source release.
Worker SHA-256: `37be65e90c3de1349fba9905a46deba1d0d1a9d12fe554e02f289a7905cf400d`.
Downloaded results: `/tmp/lunaflux-activation-readback-20260915-r2-results.tar.gz`,
local/remote SHA-256:
`3f229c6d56272628971838ecf1ee5e27a487fe28d8694bcea81c029843bbaeb9`.

Linux diagnostic worker release build passed with warnings denied. Installer
and parser regressions passed. Current local native suite: 3800/3800, native
check, format and interface generation passed. These checks do not substitute
for the remaining current-source uninstrumented campaign.

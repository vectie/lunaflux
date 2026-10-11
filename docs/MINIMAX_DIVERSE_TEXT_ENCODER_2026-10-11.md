# MiniMax original-text encoder corpus — 2026-10-11

This extends the previous three-token caption and mostly repeated 140-token
input to twelve existing original-tokenizer cases. It measures original-weight
encoder numerical differences, not full-H3 audiovisual quality, acceptance or
serving throughput. No production arithmetic, CUDA lowering or runtime hashing
was changed for the original baseline. The later normalization correction is
reported separately below; the original tensors and reference remain unchanged.

## Source-selected normalization correction

Pushed source `49b15d75` selects compensated F32 square summation explicitly in
the pure precision plan. Both row and Q/K-head normalization consume that law
at CUDA lowering; BF16 normalized-value and weight-product boundaries remain.
All twelve fresh exact-row AOT modules execute fifty layers on Spark `.179`,
using the committed ARM binary, original weights and the same literal tokens.
Every unit exits zero with finite complete final tensors and releases its GPU.
Inputs are not padded/truncated and old outputs are not overwritten.

| Input | Original worst final-row relative L2 | Compensated worst final-row relative L2 |
| --- | --- | --- |
| Actual caption | 1.1972% | 0.9558% |
| Long input | 22.1058% | 9.0533% |
| English | 1.7530% | 3.5986% |
| CJK/multilingual | 1.5909% | 1.5856% |
| Normalization | 1.8529% | 1.6672% |
| Digits | 14.7532% | 3.1238% |
| Whitespace | 1.1383% | 1.1193% |
| Emoji | 1.7026% | 1.5603% |
| Special-token text | 1.2689% | 1.2104% |
| Code | 1.2131% | 1.6576% |
| Scripts | 2.2873% | 3.1358% |
| Punctuation | 2.0948% | 2.1491% |

Eight worst-row comparisons improve and four worsen against the unchanged
library reference. Compensation fixes loss of small contributions, as the
independent component F64 oracle confirms; it does **not** establish uniform
fifty-layer reference alignment. The isolated interventions below distinguish
their interaction from a single faulty head or row kernel.
No acceptance threshold is relaxed to call this corpus a numerical pass.

Fresh source-selected digits and long outputs reproduce the separate AOT
experiments bitwise. Final tensors, per-token comparisons and summary remain at
`/tmp/lunaflux-minimax-source-norm-20261011-v2`. The comparator regression ensures
that worst-token errors are not hidden by magnitude-weighted global norms.

### Independent row/head interventions

These fifty-layer Spark runs preserve the original projections, attention,
weights, literal IDs and BF16 boundaries, changing only the named reduction.
The independent reference remains unchanged. Each number is the **worst final
token row**, which need not be the same row between runs.

| Input | Original | Row-only compensation | Head-only compensation | Both, source-selected |
| --- | --- | --- | --- | --- |
| English, 20 rows | 1.7530% | 1.6158% | 1.4635% | 3.5986% |
| Digits, 56 rows | 14.7532% | 14.2352% | 22.7025% | 3.1238% |
| Long, 140 rows | 22.1058% | 5.3483% | 21.9356% | 9.0533% |

English improves under either isolated change, but worsens under their
combination. Digits improves strongly only under their combination; changing
the head reduction alone worsens it. Therefore the earlier row-only digits
result does **not** isolate head normalization as the sole cause. The observed
fifty-layer response is nonlinear and input-dependent; the two interventions'
effects cannot be added or used to select a prompt-specific production rule.
The component F64 accuracy result remains valid, but is distinct from matching
the propagated output of a particular BF16 library implementation.
The long head-only intervention barely changes its original worst row; the
row-only intervention accounts for substantially more of that case's response.
Neither observation supplies a universal production policy: digits responds
in the opposite way to their combination.

Final tensors are under
`/tmp/lunaflux-minimax-text-encoder-compensated-{row,head}-{english,digits}-20261011-v1`;
the completed long row-only capture uses the corresponding `row-long` directory.
All completed interventions exit zero, have finite tensors and release the GPU.
The final `head-long` run is retained under
`/tmp/lunaflux-minimax-text-encoder-compensated-head-long-20261011-v2`.
Its 140 IDs match the prior row-only input byte-for-byte. This final ablation
runs on idle `.179` concurrently with the standalone full-media request on
`.178`, with independent GPUs, memory limits and no peer inference traffic.
All earlier corpus runs remain serial; concurrent lifetimes are not used as
performance measurements. Two incomplete local preparation attempts precede
`v2` (missing original manifest/input pathname); neither executes inference or
overwrites an earlier artifact.

## Method and scope

- Native ARM executable: committed `42b007d54ee6a9503cd1dfb70f570d8e23026d10`,
  freshly built on Spark `.178`; inference runs serially on Spark `.179`.
- Inputs: the existing independently matched tokenizer corpus at
  `/private/tmp/lunaflux-reference-minimax-corpus-20261011-v1`. Binary I32 input
  and decimal token IDs must agree exactly. There is no row padding, truncation
  or substitution of repeated text for diverse cases.
- Each token count gets an exact-row `sm_121` text AOT module. CPU-only export
  and compilation use 2-GiB/no-swap user units. Export consumes the original
  inventory's declared model label, not a newly computed configuration hash.
- Native execution uses the existing 60-GiB device budget and an
  80-GiB/no-swap user unit. The independent MoonBit/ATen encoder uses an
  8-GiB/no-swap container and a 2-GiB retained-tensor budget. GPU workloads do
  not overlap; the previous owner must release before the next execution.
- Reference driver/library reuse:
  `/tmp/lunaflux-minimax-aten-reference-gate-input-20261011-v1`. The driver and
  its original-weight fifty-layer implementation are unchanged.
- Both implementations retain embedding and every layer output: 51 BF16 tensors
  per case. The offline comparator reports global and individual token-row
  errors. The corpus summary includes the worst row at the final layer and the
  worst row over all layers, not only a magnitude-weighted global norm.

The diverse cases cover ordinary English, CJK/multilingual text, normalization,
numbers, whitespace, emoji, special-token text, code and punctuation. They are
not all descriptive media captions, and they do not measure caption fidelity.

## Reproduction and artifacts

Run the offline MoonBit script in three distinct phases:

~~~sh
moon run scripts/run-minimax-diverse-encoder-corpus.mbtx prepare ORIGINAL_TOKENIZER_CORPUS NEW_CORPUS_ROOT
moon run scripts/run-minimax-diverse-encoder-corpus.mbtx run ORIGINAL_TOKENIZER_CORPUS NEW_CORPUS_ROOT
moon run scripts/run-minimax-diverse-encoder-corpus.mbtx summarize ORIGINAL_TOKENIZER_CORPUS NEW_CORPUS_ROOT
~~~

For this execution, `NEW_CORPUS_ROOT` is
`/tmp/lunaflux-minimax-diverse-20261011-v2`. Its per-case receipts retain native,
reference and 51-layer comparison results without overwriting older outputs.
The initial `v1` export failed before model loading because the new harness
omitted the declared model label. Its failure receipt is preserved; `v2`
supplies that label without restoring a hashing fallback.

The summary is generated as `CORPUS_SUMMARY.tsv`. A named case can also be
summarized independently by supplying its name as the final argument; that
produces a separate per-case summary, not a complete-corpus claim.

## Completed corpus measurements

All twelve inputs complete fifty native and fifty reference layers: 612 state
comparisons, including embeddings. Every embedding matches bitwise; first
bitwise divergence is layer 1 in every case. All collected tensors have their
exact input-dependent extents and no nonfinite values.

| Input | Tokens | Final global relative L2 | Worst final token row relative L2 |
| --- | --- | --- | --- |
| Actual caption | 3 | 0.0340% | 1.1972% |
| Mostly repeated long input | 140 | 0.9633% | 22.1058% |
| English | 20 | 0.0878% | 1.7530% |
| CJK/multilingual | 28 | 0.0963% | 1.5909% |
| Normalization | 13 | 0.0735% | 1.8529% |
| Digits/Unicode digits | 56 | 0.5081% | 14.7532% |
| Whitespace | 10 | 0.4400% | 1.1383% |
| Emoji | 24 | 0.1074% | 1.7026% |
| Special-token text | 10 | 0.1281% | 1.2689% |
| Code | 38 | 0.1048% | 1.2131% |
| Scripts | 35 | 0.1123% | 2.2873% |
| Punctuation | 31 | 0.1235% | 2.0948% |

This is not confined to repeated text: digits has another substantially larger
row difference. Ten other cases have worst final-row errors of approximately
1.14–2.29%. These observations describe this corpus, not a universal bound or
an acceptance threshold. Over all layers, the long input reaches 24.5878% at
layer 48/row 5; digits reaches 17.2929% at layer 48/row 23.

The fresh no-hashing native binary's 140-token final output matches the older
diagnostic binary's final output bitwise. Removing loading/association work did
not change this case's arithmetic, but did not fix its numerical difference.

## Digits layer-46 same-input attribution

Digits row 23 increases from 5.9087% relative L2 at hidden-45 to 15.3543% at
hidden-46. Two complete independent layer-46 replays separate local arithmetic
from earlier propagated input. Reference layer 46 on reference hidden-45
reproduces every propagated reference hidden-46 BF16 value bitwise.

| Comparison, row 23 | Relative L2 |
| --- | --- |
| Native hidden-46 vs reference layer 46 on the same native hidden-45 | 0.1885% |
| Same reference layer, different native/reference hidden-45 inputs | 15.3082% |
| Whole-prefix native/reference hidden-46 | 15.3543% |

The two reference replays use identical weights and arithmetic. Their row-23
operator differences show where changed input amplifies:

| Reference operator frontier | Relative L2 |
| --- | --- |
| Attention output | 9.2189% |
| MLP normalized input | 9.4653% |
| SiLU/gate/up activation product | 41.5133% |
| Down projection | 41.8831% |
| Layer residual output | 15.3082% |

These norms are not additive. This result rules out treating the large
layer-46 jump solely as a local kernel discrepancy, but does not prove every
earlier operator correct or establish audiovisual quality. Next investigate
the earlier propagated differences and independent media quality; do not
change production arithmetic just to imitate a lower-precision reference.

Replay roots are the digits reference directory's
`replay-digits46-native-input` and `replay-digits46-reference-input`. Their
commands are reproducible with `scripts/replay-minimax-encoder-layer.mbtx`,
56 rows, hidden frontier 46 and the respective native `layers` or reference
`encode` input directory. Six CPU reports under the corpus root retain local,
propagated, attention, MLP-normalized, activation and down comparisons.

## Lifecycle and test results

All twelve native units exit 0 with zero unit swap peaks. All twelve reference
containers exit 0 without OOM, report zero retained tensors and release GPU
ownership. Both subsequent layer replays also exit 0 without OOM and release
their owners. Workloads were serial. Enforced memory limits are not measured
peaks, and native cgroup host-memory peaks do not measure all CUDA unified-memory
residency. No matched serving-throughput result follows from these lifetimes.

Affected offline script tests pass 11/11, including a 1,024-row/5,242,880-value
CPU report regression. Full per-row metrics stay in the output file; the console
now prints only global metrics, avoiding the caller's 64-KiB capture limit at
wider shapes. That CPU capacity regression is not a physical 1,024-token encoder
test. The local no-hashing check covers 34 scripts and 21 execution packages;
the token-step scan/copy/readback check also passes. No new startup prerequisite,
runtime dependency, production hash or CUDA kernel change is introduced.

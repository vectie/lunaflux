# DeepSeek matched original layer-stream replay — 2026-10-11

The bounded offline bridge calls unmodified original `Block.forward`, loading
one complete layer at a time. It retains that layer's real KV state, all four
residual streams and the literal 144 prompt IDs. After each layer, it stores
every row before retiring the original layer and weights. No original kernel
is replaced by LunaFlux and no reference dependency enters production.

Two complete 43-layer TP1 runs finished on Spark `.179`, exited zero without
OOM, and released CUDA owners:

- `/tmp/lunaflux-deepseek-original-layer-stream-full-20261011-v1`: 144 one-row
  submissions per layer.
- `/tmp/lunaflux-deepseek-original-layer-stream-prefill-20261011-v1`: one
  144-row prefill submission per layer.

The initial embedding agrees bitwise with the earlier original four-stream
capture. Full hidden tensors are 4,718,592 bytes per boundary. Captured means
refer to the final literal prompt position (143), not padded rows or generated
token frontiers. The two runs preserve both original submission modes; the
compressor only supports a full initial prefill or later one-row increments,
so the bridge does not pretend it supports six-row continuation chunks.

## What these controls show

| Matched position-143 state, layer 40 | Relative L2 |
| --- | --- |
| Native PP2, physical rows 6 versus original TP1 incremental | 21.1670% |
| Native PP2 versus original TP1 full prefill | 20.9906% |
| Original TP1 incremental versus original TP1 full prefill | 14.5487% |
| Original TP1 incremental versus original TP2 incremental | 17.2535% |
| Original TP1 full prefill versus original TP2 incremental | 17.2087% |

Native versus original TP1 incremental at layers 41/42 remains
21.0974% / 17.7848%. All compared tensors are finite. These differences are
**not additive** and do not demonstrate that the native gap is acceptable or
fixed. The older TP2 sampler reference also submits one logical row at a time:
the incremental TP1-versus-TP2 comparison controls submission shape. Comparing
TP1 full prefill to that TP2 reference changes both topology and submission
shape, and must not be described as an isolated topology comparison. Actual original
arithmetic also varies considerably across these configurations; a single
retained-state norm cannot identify a production bug by itself.

Neither replay runs the prediction heads or proves generation quality.
The two earlier agreeing greedy tokens remain a separate, narrow observation.
Docker/unit limits are 16 GiB with no swap; they are not a measured complete
CUDA-residency peak. Compiler ThreadSync warnings are retained separately.

## Later-layer matched hyper-connection control

The component runner now accepts a literal predecessor and its actual layer
index. Layer 40 receives `hidden-40.bf16` from the original incremental replay,
including four **different** streams; it does not substitute repeated embeddings.
The original `Block.hc_pre`/RMSNorm and production native renderer consume
identical original checkpoint weights and input bytes.

| Boundary | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Collapsed residual | 589,235 / 589,824 | 0.001820752% |
| Normalized branch input | 589,238 / 589,824 | 0.003406400% |
| F32 post controls | not a BF16 comparison | 0.000170275% |
| F32 Sinkhorn combination | not a BF16 comparison | 0.0000675411% |

These small local differences do not explain or eliminate the accumulated
18–21% state gap. The original and native runs terminate; memcheck/leak,
racecheck and synccheck all report zero errors, with no leaked allocations.
Root: `/tmp/lunaflux-deepseek-original-hyper-layer40-20261011-v1`.

## Next executable comparison

`cmd/deepseek_checkpoint` now has an offline `replay-layer` diagnostic. It uses
the existing production `DeepSeekDecoderSlice` for one base
layer, feeds literal original predecessor streams in explicit row batches,
and captures each complete output only after retirement. It does not change
the serving graph, precision laws or acceptance tolerances. Real matched
complete-layer replays now complete on Spark `.179`, using original TP1
incremental predecessor tensors and production native six-row batches:

| Layer | All-row relative L2 | Position-143 relative L2 |
| --- | --- | --- |
| 3 (compression ratio 128) | 0.2316441% | 0.6626280% |
| 40 | 0.0037391% | 0.547154% |

Both runs exit zero and release CUDA owners. Layer 40's small global norm is
dominated by unusually large early-row features; it is not evidence of equally
small error on every row. Layer 3 includes individual rows exceeding 4%.
These controls separate local complete-layer error from the accumulated gap,
but do not isolate submission geometry or prove full-model parity.
Roots: `/tmp/lunaflux-deepseek-production-layer-3-rows6-20261011-v1` and
`/tmp/lunaflux-deepseek-production-layer-40-rows6-20261011-v1`.

First-three-layer replay now accepts literal original I32 token IDs and loads
the actual checkpoint token-to-expert tables through the production index-table
uploader. It does not substitute learned routing for token-hash routing. Its
committed ARM executable now completes the first layer using one-row submissions
and all 144 original IDs. Layer 0 aggregate relative L2 is **0.3749400%**;
row 47 is **1.6313947%**. This is the first matched complete-layer discrepancy,
not only a late-layer accumulated effect. Root:
`/tmp/lunaflux-deepseek-production-layer-0-rows1-20261011-v1`.

The next diagnostic records attention input/output, the complete four-stream
attention residual, and FFN input/output after native queue retirement. An
optional original CPython return observer records the same boundaries from
unmodified `Block.forward`, RMSNorm, Attention and MoE, retaining one-row
submissions and token-hash routing. Ordinary serving has no observer, readback,
extra launch or changed arithmetic. The literal one-row layer-0 boundary run
now completes, with all 144 actual token IDs:

| Boundary | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Attention input | 589,824 / 589,824 | 0% |
| Attention output | 491,296 / 589,824 | 0.1384427% |
| Four-stream attention residual | 2,298,899 / 2,359,296 | 0.0932574% |
| FFN input | 530,322 / 589,824 | 0.1506909% |
| FFN output | 156,579 / 589,824 | 0.5006931% |

Original root: `/tmp/lunaflux-deepseek-original-layer-stream-boundaries-20261011-v1`.
Native root: `/tmp/lunaflux-deepseek-production-layer-0-boundaries-20261011-v1`.
The first normalized branch input is bitwise identical; divergence first
appears inside attention. The larger FFN discrepancy includes a different
input, so it cannot yet be attributed solely to routed experts. A matched FFN
input intervention is required. The full-model numerical difference is **not
fixed** by adding an observer.

## Literal original FFN row-47 intervention

The layer-0 row with the largest complete-layer relative error is replayed using
the original `ffn-input.bf16` row **47** and its actual original token **45948**.
It does not use the changed native FFN input or substitute BOS routing.
All six original hash-selected routed experts complete on `.179`; both native
matched-down-input and propagated gate/up/down modes agree **bitwise** at every
weighted activation and output (12,288 activation values and 24,576 output values
per mode). Original/native runs exit zero and release CUDA owners.

Root: `/tmp/lunaflux-deepseek-original-expert-row47-20261011-v1`.
This excludes those six routed-expert kernels as the local source of this row's
discrepancy. It does not test the shared expert, final MoE combination, other
rows, or full-model parity. Attention remains the first differing branch.
The retired observer now exposes rotary query/KV, sparse attention and grouped
output-A as well, so that production one-row execution can be compared without
confounding it with the earlier 144-row component geometry.

```sh
moon run scripts/run-deepseek-original-layer-stream.mbtx prepare NEW_ROOT 43 [--prefill]
moon run scripts/run-deepseek-original-layer-stream.mbtx run ROOT 43
moon run scripts/run-deepseek-original-layer-stream.mbtx collect ROOT 43
moon run scripts/replay-deepseek-original-hyper.mbtx prepare NEW_ROOT IDS_TEXT --attention-layer 40 ORIGINAL_HIDDEN_40_BF16
```

Four hyper-runner regressions and the streamed-runner regression pass.
Warning-denied native checking and the full suite pass **5,015/5,015**, with
the existing toolchain migration-warning exclusions and no added exclusions.
Offline reference bridges compile with `-Wall -Wextra -Werror` on ARM.

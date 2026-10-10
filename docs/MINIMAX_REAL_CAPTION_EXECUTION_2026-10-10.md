# MiniMax real caption execution — 2026-10-10

Source: `b07c031313ef858ed63aedf6f1638857fd8ea907` on `main`, including
the preceding omitted-BPE-`ignore_merges` fix `d92f49fb`. This is functional
checkpoint execution, not an optimized throughput benchmark or independent
upstream numerical parity result.

## Literal text frontend

The original MiniMax text encoder tokenizer JSON encodes `A red cat` into
`32,2518,8251`. The new shared native checkpoint frontend adds no BOS/EOS or
chat template, applies the declared normalization/added-token semantics, and
serializes the three IDs as little-endian I32. Decoding returns the original
caption and stderr is empty. Additional CPU cases cover special tokens,
multilingual NFC normalization, context overflow, empty captions and visual
placeholders without placement. Existing GLM input retains all 13 prompt IDs
and all 13 execution frames.

These are caller-supplied tokenizer labels, not computed checksums. No weight
inventory, shard payload authentication or CUDA work is performed by `text-ids`.
Related native tests passed 47/47; expanded tokenizer/frontend tests passed
98/98. The 26-script no-hashing regression passed.

### Independent original-tokenizer corpus — 2026-10-11

The shared native text frontend now matches vLLM 0.26.0's HF renderer on
twelve original text-encoder tokenizer cases, for both exact IDs and exact
decoded text. The actual `A red cat` caption still emits `32,2518,8251`.
The remaining cases cover a 140-token literal prompt, English, CJK/kana/Hangul,
NFC/decomposed Unicode, mixed numeric scripts, whitespace/CRLF, emoji, original
text special tokens, source code, multilingual scripts and punctuation.
Reference counts in that order are `3,140,20,28,13,56,10,24,10,38,35,31`;
all 24 comparisons pass. Both sides disable implicit special-token insertion.
Image/video placeholder placement is deliberately not part of this text-only
corpus and remains owned by the multimodal presentation path.

The existing H3 ARM image uses Transformers 5.14.1 and tokenizers 0.22.2.
`vllm launch render` reads the original, unmodified Qwen3-VL text-encoder
configuration/tokenizer, not denoiser weights. Its loopback-only container has
4 GiB/no-swap limits; measured memory peak is 2,452,099,072 bytes and swap peak
zero. It exits successfully after collection, and its GPU compute-owner query
is empty. This is independent frontend parity on the stated corpus, not hidden
state/logit equivalence, encoder throughput or media quality.

Evidence: `/private/tmp/lunaflux-reference-minimax-corpus-20261011-v1`;
startup: `/private/tmp/lunaflux-reference-minimax-20261011-v1`.
Native outputs, original text, raw reference responses and terminal/resource
observations are retained without payload hashing or overwriting older runs.

## Real GPU text encoder

The exact committed source was built on .178 with the installed ARM toolchain:
184 build tasks, exit 0, peak host/cgroup memory 665.8 MiB, swap peak 0.
The executable then ran on .179 against original streamed text weights and
the existing three-row request `text.cubin`:

| Item | Observed |
| --- | --- |
| Input | Actual caption token IDs `32,2518,8251` |
| Device | Spark .179, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6` |
| Packed weight allocation | 51,506,191,840 bytes |
| Request allocation budget | 357,916 bytes |
| Output | `hidden_states[50]`, BF16 `[3,5120]`, 30,720 bytes |
| Output statistics | Nonfinite 0; maximum absolute value 15,168 |
| Process | Exit 0; GPU compute owner released |
| Host/cgroup accounting | Journal reports peak 8.6 GiB, swap peak 0 |
| External limits | `MemoryMax=80G`, `MemorySwapMax=0` |

Host/cgroup accounting is not the same as driver-reported GPU allocation;
the unified-memory device budget remains independently bounded at
64,424,509,440 bytes. The output was downloaded without overwrite and has the
expected byte count. Finite output alone does not establish numerical parity;
the maximum absolute value is reported rather than hidden or interpreted as
proof of quality.

Local evidence:
`/tmp/lunaflux-minimax-text-encoder-caption-20261010-v1-179`.
Original frontend evidence:
`/private/tmp/lunaflux-minimax-caption-20261010.KQkB7Fiu` and
`/private/tmp/lunaflux-checkpoint-text-live-20261010.FwEAvaU2`.

### Independent original-weight encoder comparison — 2026-10-11

The actual caption now has an independent original-weight arithmetic reference,
not just finite output or an internal component oracle. A separate test-only
MoonBit module in `benchmarks/checkpoint_aten_reference` reads the original
safetensors metadata and selected weight regions directly. It owns layer order
and explicit tensor scopes; a private native bridge calls the ATen library
already installed in the H3 reference image. The production module imports
neither this diagnostic nor PyTorch. No Python automation, payload hash, or new
startup prerequisite was added.

The implementation follows the installed upstream
`vllm_omni/diffusion/models/minimax_h3/encoder.py` text-only math: three separate
Q/K/V BF16 linears, Q/K RMSNorm, staged BF16 rotary products/sum, causal GQA SDPA,
output projection, residuals and separate gate/up SiLU MLP. It retains exactly
the first 50 decoder layers, without the language-model final norm. It does not
reuse LunaFlux's packed weights, rotary tables or CUDA kernels. The installed
ATen context enables cuDNN SDPA by default, matching the source's explicit
enable. This is an independent ATen execution of that arithmetic, not an
execution of the complete upstream Python H3 pipeline.

Both fresh .179 runs (`v2`, then final-driver `v3`) complete all 50 layers for
`32,2518,8251`, write every layer's BF16 output and release all retained tensor
owners. Their final 15,360 BF16 values are bitwise identical to each other.
Compared with the previously executed native `b07c0313` caption encoder:

| Metric | Measured |
| --- | --- |
| Exact BF16 values | 4,073 / 15,360 |
| Nonfinite on either side | 0 |
| Mean absolute error / RMSE | 0.0240924 / 0.0420121 |
| Absolute-error p50 / p95 / p99 | 0.015625 / 0.078125 / 0.1328125 |
| Global relative L1 / L2 | 0.6528% / 0.0340% |
| Largest absolute error | 1, at native 176 versus reference 177 |
| Maximum absolute value on both sides | 15,168 |

Global cosine similarity is 0.9999999422, but large-magnitude channels dominate
that aggregate. Report the actual token-row differences rather than using it
alone as an equivalence claim:

| Caption row | RMSE | Relative L2 | Cosine |
| --- | --- | --- | --- |
| 0 | 0.0165571 | 0.00774% | 0.9999999970 |
| 1 | 0.0380258 | 0.78214% | 0.9999699130 |
| 2 | 0.0597909 | 1.19722% | 0.9999374821 |

There is no invented model-specific tolerance or automatic pass based on these
numbers. They establish a reproducible text-encoder differential for this
caption; broader inputs, exact reduction-level attribution, joint-denoiser/VAE
reference comparisons and perceptual caption fidelity remain open.

The final reference container exits 0 with `OOMKilled=false`, configured
memory and memory-plus-swap limits both 8,589,934,592 bytes. Live reads of its
actual Docker process cgroup report a peak at that 8-GiB ceiling and swap
current/peak zero. This includes file-cache/workspace effects and is **not** the
tiny systemd Docker-client wrapper's memory figure. The SDK retains at most
2 GiB of explicitly owned tensor storage, uploads one projection region at a
time and ends with retained bytes zero. The peak touches the external limit,
so this is not a claim of spare memory or safe scaling to longer requests.
Both GPU-owner queries after collection are empty. The 40.460/44.167-second
reference unit lifetimes include loading, are two diagnostic samples, and are
not a matched serving or vLLM performance comparison.

Validation: standalone native warning-denied check and three tests pass without
warning exclusions; two ARM CPU smokes pass BF16 normalization, short-read and
oversized-allocation rollback, closed operand and deterministic release. The
final CPU smoke also checks metadata-read recovery after a native error. The
ATen C++ ownership probe passes Linux ASan/LSan with empty stderr; its native
loader/metadata tests also pass macOS ASan (macOS leak detection disabled).
The supplied skill's older ASan helper could not patch the current `moon.pkg`
syntax; its changes were restored and the loader tests were rerun from a
separate instrumented scratch module instead. No production toolchain or
package flags were changed.

Final evidence: `/tmp/lunaflux-minimax-aten-reference-20261011-v3`, including
`encode/hidden-0.bf16` through `hidden-50.bf16`,
`hidden-50-comparison.txt`, `reference-repeat-comparison.txt`, terminal receipts
and actual cgroup samples. Earlier reference/ASan evidence is preserved in
`/tmp/lunaflux-minimax-aten-reference-20261011-v2`; the first source archive's
macOS AppleDouble build failure remains under `v1`. All outputs are retained
without payload hashes or overwriting previous experiments.

## Joint request

The actual caption hidden state is now the input for the separate .178 joint
request. It uses the existing AOT geometry (32×32, 120 frames, 5 denoising
evaluations), not a realistic-resolution quality or performance test. The
encoder weight owner has exited before denoiser preparation; its 51.5 GB
allocation is not kept alongside the denoiser weights.

The exact user unit is
`lunaflux-minimax-joint-caption-20261010-v1.service`; local/remote evidence root
is `/tmp/lunaflux-minimax-joint-caption-20261010-v1`. The request terminated
successfully at 23:54:40 CST after starting at 23:24:58 CST: 1,782 seconds
including preparation/loading and execution, not steady-state GPU-only latency.
The retained unit reports `Result=success`, `ExecMainStatus=0`, `MainPID=0` and
`SubState=exited`. Its peak host/cgroup memory is 51,214,577,664 bytes. Observed
running samples reported zero cgroup swap; a terminal swap peak is unavailable
and is not inferred from those samples.

Terminal stdout reports five completed denoising evaluations, 50 denoiser
layers, both decoded outputs, 165,600 audio samples and zero nonfinite values.
Stderr is empty. The terminal collector observed an empty GPU compute-owner
query before dispatching the subsequent DeepSeek test; no encoder or joint
owner was kept alive alongside that model.

Both raw F32 files were downloaded without overwrite to
`/tmp/lunaflux-minimax-joint-caption-download-20261010-v1`, and independently
inspected offline with `scripts/collect-minimax-joint-output.mbtx`:

| Output | Bytes / F32 values | Nonfinite | Nonzero | Maximum absolute | Mean square |
| --- | --- | --- | --- | --- | --- |
| Video | 1,523,712 / 380,928 | 0 | 378,502 | 1 | 0.1264938080 |
| Audio | 1,324,800 / 331,200 | 0 | 331,200 | 0.2681674361 | 0.0012207769 |

This establishes actual-caption end-to-end component execution and complete,
finite, nontrivial decoded arrays at the stated small geometry. It does not
establish independent reference equivalence, caption fidelity, perceptual
quality, realistic-resolution performance or a production serving result.

## Output-length contract correction — 2026-10-11

The requested 120 video frames resolve to 124 by the declared temporal shape
law. At 32×32 RGB F32 this is exactly 1,523,712 bytes. The AudioVAE retains
207 rounded latent frames × 800 samples = 165,600 samples per channel,
exactly 1,324,800 stereo F32 bytes. This is the reference's latent-rounded
duration, not accidental trailing output to crop away.

An integration mismatch remained: generic result metadata and reference output
budgets still used the nominal frame-duration count, 165,333. The shared pure
shape contract now explicitly chooses `FrameDuration` or `LatentStride(stride)`;
MiniMax selects the latter. `audio_samples()` remains nominal timing, whereas
`audio_output_samples()` determines publication length. Canonical results,
reference output budgets and prepared decoder joins use the actual extent.
The rule is selected in the model adapter, not a family branch in generic
result handling or CUDA dispatch. It adds no payload hashing or token-step work.

The existing physical output above predates this metadata correction. No decoder
arithmetic or media bytes were changed, and no new GPU execution or performance
claim follows from changing their result contract.

The affected nineteen-package native regression matrix passes 235/235, including
the default duration law, rounded output counts, invalid strides, exact stereo
output-budget boundary, terminal metadata and prepared decoder joins. Existing
CUDA source-byte snapshots remain unchanged; plan-dependent recipe snapshots
were intentionally refreshed for the new output law. Whole-repository native
check and affected formatting/interfaces pass with the existing migration
warning exclusions `-20-25-29-35-79-92`, not an unfiltered warning-clean claim.
The 27-script no-hashing and token-step scan/copy/readback checks also pass.

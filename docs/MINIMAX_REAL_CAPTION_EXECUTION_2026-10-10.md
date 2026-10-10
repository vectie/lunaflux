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

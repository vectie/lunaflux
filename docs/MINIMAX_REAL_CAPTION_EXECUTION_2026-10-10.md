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
is `/tmp/lunaflux-minimax-joint-caption-20261010-v1`. This section records
dispatch only until terminal output, both decoded files, release and memory
results have been inspected. Independent reference equivalence and realistic
media quality remain unverified.

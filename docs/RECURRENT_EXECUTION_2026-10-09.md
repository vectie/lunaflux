# Request-owned recurrent delta execution

The executable slice is the normalized delta recurrence, not a complete GLM
attention block or a whole-model benchmark. Pure precision IR describes BF16
frame operands, F32 log decay/state, geometry, numerical ordering and state
budget. The GLM adapter selects that plan only for KDA layers. CUDA lowering
owns thread geometry; the Program owns cache lifetime and submission effects.
Model names and CUDA branching are absent from the generic plan/executor.

Slot IDs address persistent requests independently of CSR batch order. A slot
can continue after compaction/reordering or reset for a new request. A lane owns
one value column's ordered key fold; causal token order is retained. Cache is
read/written at frame boundaries. Startup owns allocation, zeroing, AOT binding
and queue creation; token steps perform neither allocation nor state readback.

## Validation and bounded component timing

Idle .178 GB10, UUID `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`, CUDA 13.0.88,
`sm_121`, `--fmad=false`. Each command has a 2 GiB host-memory cap, no swap and
120-second limit. Tests retain at least 32 GiB available memory. Final host
snapshot: about 4.2 GiB used / 117 GiB available, with no compute process.

| Heads × keys × values / rows | Serial reference median | New median | New range, five pairs |
| --- | ---: | ---: | ---: |
| 2 × 4 × 6 / 4 | 22.71 us | 4.13 us | 4.00–4.26 us |
| 4 × 32 × 16 / 8 | 1,046.94 us | 6.20 us | 6.07–6.25 us |
| 64 × 128 × 128 / 8 | 678,741.19 us | 47.63 us | 42.67–48.13 us |

The reference deliberately uses one thread, scans inputs and publishes state
per token. Those large ratios do not establish a production speedup or parity
with vLLM/SGLang. The experiment measures two recurrent component schedules.
The new GLM shape still consumes 255 registers, with 276-byte spill stores and
280-byte spill loads reported by ptxas. Register pressure remains future work.

All three shapes match reference output and active F32 state bitwise in ordinary
runs, and pass an independent double-precision oracle through continuation,
slot reorder, reset, unequal dimensions and idle frames. Maximum observed oracle
error is 9.54e-7. New-kernel memcheck reports zero errors/leaks; racecheck reports
zero hazards; synccheck reports zero errors. Sanitizers use the independent
oracle without running the deliberately serial reference: its full-size
instrumented memcheck exceeded the 120-second cap. That timeout and an earlier
invalid `-5f` reference compile are preserved, not relabeled as passes.

Thirty-six focused native tests pass. The queue regression covers 32 steps with
zero measured heap allocations/blocking waits, cancellation and partial prepare
cleanup. Native interfaces/format pass. Warning-denied checks fail on existing,
unrelated deprecations; their migration remains paused.

## Artifacts and next composition

Remote artifacts: `/tmp/lunaflux-recurrent-feature.4Ynf8F` on .178.
Local capture: `/tmp/lunaflux-recurrent-capture.ISWTIv`.
Downloaded header/probe/binary hashes match the remote files. The non-overwriting
archive `/tmp/lunaflux-recurrent-capture.ISWTIv.tar` has SHA-256
`4a4e7f0ed7b335bf7b51f8ae781b7cc110296bbcb3502b546c72aa86b27e48ab`.

The complete block still needs checkpoint projections, decay/input gates,
output gating/norm/projection and worker
composition. DSA, DeepSeek compressed attention, MiniMax full execution and
two-host model execution remain unfinished. No TLS/admission expansion was added.

## Q/K/V convolution composition added

The Program now prepares three independent request-owned BF16 convolution
histories and three intermediate buffers, then submits Q/K/V convolution and
delta update in order with one completion event. One source symbol is reused
with three parameter/history bindings. History contains raw projected inputs,
not SiLU outputs. Request slots, resets and CSR offsets are shared across the
four kernels; idle state and inactive rows/slots remain unchanged.

The generic precision IR plans exact state/frame bytes. CUDA lowering uses
independent channel lanes and no workgroup barriers. Model-specific geometry
stays in the GLM adapter. Startup compiles both AOT symbols and binds them once;
no request JIT, state readback or added admission layer is involved.

The updated focused native suite passes 42/42, including 32 composed steps /
128 launches with zero measured heap allocations and blocking waits. Partial
preparation releases all owned histories; active cancellation drains before
release. Ordinary native check/interface generation and formatting pass.
Warning-denied checking still fails on 30 existing unrelated deprecations.

GB10/CUDA/tool and per-command memory/runtime limits are unchanged from above.
The GPU test executes the full four-kernel component chain, not merely source
rendering. All three shapes pass the independent double oracle, bit-exact raw
history, continuation/reorder/reset/idle and one eight-token prefill versus eight
single-token decode frames with bitwise-identical final output and all states.

| Channels / convolution kernel | Recurrent geometry | Maximum oracle error |
| --- | --- | ---: |
| 8 / 2 | 2 heads × 4 dimensions | 5.23e-9 |
| 129 / 4 | 1 head × 129 dimensions | 8.30e-9 |
| 8192 / 4 | 64 heads × 128 dimensions | 1.53e-5 |

Memcheck reports zero errors/leaks, racecheck zero hazards and synccheck zero
errors on all three chains. Convolution kernels use 29–31 registers, no spill
traffic or barriers. The delta kernel retains its previously documented spill
pressure. This work adds missing stateful execution; it does not claim a serving
performance improvement or close full KDA/model execution.

Remote artifacts: `/tmp/lunaflux-convolution-feature.DkIqyn` on .178.
Local capture: `/tmp/lunaflux-convolution-capture.sqZumO`.
Downloaded header/probe/binary SHA-256 hashes match remote artifacts. The new
non-overwriting archive `/tmp/lunaflux-convolution-capture.sqZumO.tar` has SHA-256
`a16cc795eaa93c94df69acf97c027e498d9ec5ec21722398478570edbc534779`.

## Complete recurrent branch

The executable Program now includes the remaining numerical stages in one
startup-prepared queue:

`hidden → Q/K/V + forget/beta + output gate → controls → Q/K/V convolution
→ normalized delta/cache commit → sigmoid-gated RMSNorm → output projection`.

Sixteen launches share one completion event, not sixteen CPU waits. Generic
precision IR owns geometry, intermediate BF16 rounds, F32 decay/cache and memory
accounting. CUDA lowering reuses per-output parallel ordered dense projections
and the existing elementwise decay renderer. The GLM adapter supplies hidden
4096, heads 64, dimension/control-rank 128, convolution four, lower decay -5 and
the model's output RMSNorm epsilon 1e-5. Recurrent Q/K normalization keeps its
separate 1e-6 epsilon. No model-specific branching entered the executor.

At rows 32 / sequences 16 / cache slots 32, all owned cache and intermediate
frames total 145,252,352 bytes, excluding borrowed weights/input/output. A tiny
6-hidden / 2-head / 4-dimension / rank-3 GPU fixture executes the same 16-stage
composition. Its independent double oracle reports maximum absolute error
3.60905531e-9 over ragged continuation, slot reordering and reset. Raw histories
match exactly; one eight-token prefill and eight one-token decode frames produce
bitwise-identical output and F32 state. Memcheck reports zero errors and leaks,
racecheck zero hazards and synccheck zero errors. Each remote process was capped
at 2 GiB with no swap and a 120-second timeout. Available host memory remained
123,207,988 KiB. Combined native checks pass 47/47; 32 full-branch steps produce
512 enqueues with zero measured heap allocations and blocking synchronization.
Targeted format and native interface generation pass; unrelated existing
deprecation warnings still prevent claiming a warning-denied release boundary.

The first physical probe compile found a host `round` name collision. It was
renamed to `bf_round`; the failed source/logs remain in their original directory.
No numerical kernel change was needed for this probe-only issue.

Successful remote artifacts: `/tmp/lunaflux-block-feature.Wq3BDA` on .178.
Local capture: `/tmp/lunaflux-block-capture.REIPp9`; downloaded header/probe/binary
hashes match remote. Archive `/tmp/lunaflux-block-capture.REIPp9.tar` has SHA-256
`15d9ddb65051cc512bf34a2d793d5966e7504f004f03c9d7826c957cf704536d`.

This closes the executable recurrent-branch composition gap, not full GLM
inference. Real checkpoint ownership, surrounding mHC/MLP and DSA execution,
worker wiring, DeepSeek and MiniMax full-model execution and two-host reduction
remain unfinished. Ordered projections are correctness-first parallel schedules,
not tensor-core GEMM; no serving throughput or reference-engine speedup is claimed.

## Checkpoint-backed layer ownership

`GlmRecurrentLayer` now composes exact model weight binding, the generic
`CheckpointDeviceWeights` uploader and the prepared recurrent Program. The
model adapter owns names/shapes, precision materialization preserves dtype and
physical rank, the uploader owns allocations, and the Program owns request
state/submission. Rank-three dense binding adds no transpose, conversion or
full-size host staging. Startup uses one reusable chunk for all fifteen weights.

The actual layer-0 index and header on .175 were read directly from
`/data/models/LibertAIDAI/GLM-5.3-Flash-NVFP4`. All fifteen parameter names,
dtypes and shapes match the adapter, including BF16 convolutions `[8192,1,4]`,
F32 `dt_bias[8192]` and `A_log[64]`. No checkpoint file was changed. Neither
enterprise Spark has `/data/models` mounted; that source location must not be
assumed local to the GPU worker.

A sparse official-sized native fixture avoids a full host weight arena. It
binds/uploads 275,481,088 weight bytes and prepares 145,252,352 workspace bytes
at rows 32 / sequences 16 / slots 32. Its 32 steps enqueue 512 kernels with zero
measured heap allocations/blocking waits. Active cancellation drains the
executor before releasing borrowed weights and leaves no live test resources.
Separate small BF16/F32 readback verifies exact transferred bytes across
three-byte chunk boundaries. Rank permutations with equal byte counts are
rejected instead of being silently reshaped. These are native test-double
results; executing real checkpoint weights on GPU and whole-model composition
remain open. Existing small-chain GPU numerical results above are unchanged.

No TLS, signing, registry or deployment-hardening work was added.

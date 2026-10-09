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
request-owned short convolution, output gating/norm/projection and worker
composition. DSA, DeepSeek compressed attention, MiniMax full execution and
two-host model execution remain unfinished. No TLS/admission expansion was added.

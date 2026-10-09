# Request-owned causal short convolution

The precision IR declares BF16 history and operands, F32 ordered accumulation,
SiLU and BF16 output rounding. CUDA lowering gives each lane one channel within
an explicit CSR sequence. Raw history remains lane-local through the token fold
and commits to the request slot once. Outputs never enter the history cache.

The eight-pointer ABI is counts, offsets, request slots, reset flags, input,
weights, in-place history, output. Slots must be exclusive within a frame.
History is `[slot, channel, oldest-to-newest lag]`; weights are
`[channel, oldest-to-current kernel]`. Idle and inactive slots are unchanged.
The current lowering supports kernel sizes 2–32 and tails not divisible by 128.
This is generic convolution lowering, not a GLM special case.

`RecurrentDeltaProgram::with_convolution` binds three independent instances
before recurrent update in one prepared queue. It owns all histories and
intermediates; projection/control/gating and full model execution are not added
by that composition. `source_bytes` supplies both symbols for startup AOT.

The bounded CUDA test checks independent double-precision arithmetic, exact raw
history, request reordering/reset/idle and bitwise prefill-versus-decode chunks.
It executes all four kernels on the same stream. It is component testing, not
a serving throughput benchmark.

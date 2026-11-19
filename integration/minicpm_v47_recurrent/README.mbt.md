# MiniCPM-V 4.7 recurrent branch

`Program` composes the existing matrix, causal-convolution, gated-delta-control,
recurrent-delta and gated-RMSNorm source packages. Four BF16 input projections
produce packed Q/K/V, gate z and controls a/b; convolution and controls feed the
grouped recurrence, followed by gated normalization and output projection. Its
nine entrypoints share `recurrent_branch.cu`. Input is already decoder-normalized
and residual addition belongs to the enclosing decoder.

One caller-owned scratch arena uses 1,845,248 bytes at 32 rows, reusable serially
across recurrent layers. K/V input pointers are byte-offset views into the
convolved packed QKV bank; their minimum spans preserve the 8192-BF16 row stride
without allocating or repacking. Persistent BF16 convolution history and FP32
recurrent matrices remain outside scratch. At one slot these occupy 49,152 and
2,097,152 bytes per recurrent layer. Budgets scale with slot capacity, separately
from the live sequence count.

`Frame` borrows all modules, metadata, weights and buffers, including offset views
into persistent state arenas. It owns nine function handles and prepared launch
descriptions for the existing ordered executor. Close enclosing executors before
the frame, then release borrowed resources. Close partial preparation on failure.
Caller-owned counts/CSR/request slots must be valid and slots exclusive within a
frame. Both stateful kernels use the same reset flags; reset begins a new request
and continuation retains history and matrices. No per-token state copying or
argument reconstruction occurs in this owner.

Preparation and runtime consume the same program's source, stages and regions.
CPU tests verify projection shapes, argument order, packed views, workspace and
independent slot capacity. They do not establish CUDA compilation, actual device
prepare/execute/cleanup, or upstream GPU numerical parity. Whole-model state
ownership, weight loading and decoder assembly remain separate integration work.

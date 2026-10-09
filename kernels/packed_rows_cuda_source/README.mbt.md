# BF16 packed row assembly lowering

One out-of-place kernel reads three compact BF16 source matrices through an
immutable row map and writes packed output, explicitly zeroing padding. Values
are copied as unsigned 16-bit cells, preserving NaNs and signed zero without
floating-point arithmetic. Input/output must not overlap; empty sources may
use an unread placeholder pointer. Geometry is fixed at AOT preparation.

Host tests validate mapping and emitted source, not physical CUDA correctness
or bandwidth. This kernel does not perform projection or conditioning logic.

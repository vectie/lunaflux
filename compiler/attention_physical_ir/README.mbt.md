# Online attention physical state dialect

`OnlineFold` is an immutable, backend-neutral register-state program for the
ordered online-softmax recurrence. QK produces raw scores; the value phase
applies scale/visibility and computes a local maximum, then merges
the previous maximum, rescales the previous output, forms BF16 probabilities
and accumulates PV, then updates the F32 denominator. Each operation exposes
named read/write values, and the loop carries three explicit state edges.

The generic selected schedule constructs this program before CUDA lowering.
The selected query-owned CUDA emitter consumes these operations, rather than
owning their ordering as one opaque source block. Existing schedule-owned
transfer/publication/release effects surround the two physical phases. This
does not turn attention into GEMM or add runtime compiler work.

The current numerical contract preserves ascending-key folding, BF16-RNE PV
probabilities and an unrounded F32 denominator. Device instruction realization,
lane mapping, masked exponential, and reduction topology remain backend-owned.

`KeyFold` separately describes the F32 per-key statistics merge and scale
publication used by direct, grouped and split-key attention. It computes
component ownership from an explicit backend-supplied owner width, without
changing the scalar probability law into the BF16 matrix law. The schedule's
closed `AttentionPhysicalProgram` binds these distinct dialects and the
shared-score program to their transfer/storage effects. Full device instruction
and copy-address realization remain backend responsibilities.

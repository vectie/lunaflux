# Physical CUDA validation probe

This native-only executable validates the real CUDA Driver/cuBLASLt boundary.
It exercises inventory, contexts, streams, events, allocations, fixed-buffer
round trips, exact BF16 GEMM numerics, PTX module/function loading, direct AOT
launch, capture-required startup CUDA graph execution, polling, and
deterministic reverse-order cleanup. The residual-CUBIN extension deliberately
retains the ordered eager path as a separate control.

Compile `probe.cu` to PTX for the selected GPU before running:

```sh
nvcc -ptx -arch=compute_120 tests/physical_cuda/probe.cu \
  -o /tmp/lunaflux-physical-probe.ptx
moon run --target native tests/physical_cuda -- \
  /tmp/lunaflux-physical-probe.ptx 128
```

When a deterministic residual-add CUBIN produced by the Luna CUDA AOT builder
is available, pass it as the optional fourth argument. The probe then loads the
generated symbol, enqueues the exact three-pointer ABI through the ordered
executor, waits on its completion event, and checks BF16 output bytes:

```sh
moon run --target native tests/physical_cuda -- \
  /tmp/lunaflux-physical-probe.ptx 128 \
  /tmp/kernel-root/sha256/ARTIFACT_DIGEST.cubin
```

The runner is evidence for the private CUDA primitive boundary. It does not
claim an end-to-end model, paged-attention, I8, tensor-parallel, or serving
promotion gate. The optional residual check validates only that one generated
kernel specialization.

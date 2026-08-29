# Symmetric-I8 weight+scale physical CUDA probe

This is a deliberately narrow physical proof for one LunaFlux catalog-v4
`OutputProjection`. It admits the production symmetric-I8 v1 device feature,
builds the real one-layer dense-Llama I8 plan, admits the closed production I8
policy, derives the canonical paged-v4 operand sequence, authenticates the
compiled CUBIN through the existing artifact bundle, and launches the selected
entry point through the public ordered-device API. The CUDA ABI is exactly:

`StepCounts, BF16 activation, I8 row-major weight, F32 per-output scale, BF16 output`.

The independent MoonBit referee reconstructs every weight as
`F32(code) * scale`, preserves declared F32 operation order, and compares the
BF16 result bit-for-bit. The fixture includes both `-127` and `127`; `-128` is
absent because production v1 reserves it.

This is not a full-model, loader, scheduler, serving, performance, memory, or
accuracy proof. The surrounding dense-Llama graph is admitted because current
catalog-v4 launch and artifact admission are intentionally graph-complete;
only its 4-by-4 I8 output-projection entry point is executed. The other exports
are ABI-complete inert fixtures and are never used as numerical evidence.

Production I8 v1 device admission is closed to exact compute capabilities 8.9
and 9.0. Consequently the known `sm_120` RTX 5060 Ti validation host must reject
this probe and cannot close the I8 physical gap. Run the probe only on an
observed BF16-capable `sm_89` or `sm_90` device with CUDA 13.1 `nvcc`:

```sh
scripts/probe-i8-weight-scale-cuda.sh /absolute/path/to/nvcc sm_89 32
```

Compilation happens before the MoonBit process starts and emits a CUBIN. The
runtime never accepts PTX and has no NVRTC or other JIT path.

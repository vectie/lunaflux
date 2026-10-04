# Precision representation IR and format implementation

## Architecture

This is a refinement of the existing functional compiler, not a replacement
compiler, a model-specific branch, or a new request-time JIT.

```text
Model/operation semantics
  │ logical matrix shapes, values, numeric accuracy policy
  ▼
Precision / representation IR                   compiler/precision_ir
  │ storage vs compute, scale grid, conversion placement, explicit write effects
  │ quantization: global amax → scale amax → payload publication
  ▼
Scheduled tile IR                              luna_projection_tile_compiler
  │ retained matrix/decode schedules and numerical reduction order
  ▼
Physical tile IR + precision refinement         physical_tile_ir
  │ same immutable producer/consumer layout, lifetime, fragments and ownership
  ▼
Device lowering                                luna_cuda_precision / projection_aot
    CUDA conversions, shared stores, async input copies, BF16 MMA instructions
```

`model/precision_format` owns backend-neutral numeric encodings. It imports no
CUDA, device, compiler, scheduler, or model family. `compiler/precision_ir`
contains immutable plans and pure transformations. Startup materialization
owns the explicit read/upload effects; the CUDA renderer owns hardware details.
No global runtime state or token-step filesystem/cryptographic work was added.

Four-bit packing is explicitly row-aligned, low logical column in the low
nibble, with zero high tail padding. Logical scales are row-major. A checkpoint
swizzle is not inferred from a filename or a format label.

## Implemented format semantics

| Format | Payload | Scale metadata |
| --- | --- | --- |
| BF16 / FP16 / F32 | 16 / 16 / 32 bits | None |
| FP8 E4M3FN / E5M2 | 8 bits | Positive F32, tensor/row/2D-block grouping |
| INT8 / INT4 | Signed two's complement, 8 / 4 bits | Positive F32, tensor/row/2D-block grouping |
| FP4 E2M1 | 4 bits | Positive F32, tensor/row/2D-block grouping |
| NVFP4 | E2M1, 4 bits | E4M3FN per 16 values plus one F32 global scale |
| MXFP4 | E2M1, 4 bits | UE8M0 per 32 values |
| MXFP8 E4M3 / E5M2 | 8 bits | UE8M0 per 32 values |

The integer codec uses the full signed ranges `[-128,127]` and `[-8,7]`.
It deliberately does not reinterpret legacy I8's reserved `-128` encoding.
Existing numeric-contract v1/v2 identities are unchanged.

Rounding is nearest-even with finite saturation. UE8M0 scale selection rounds
upward to a representable power of two; code 255 is not a finite scale. NVFP4
reconstruction uses two ordered F32 multiplications, local scale then global
scale. Global amax publication precedes local scale reduction and packing.

These format definitions follow the [NVIDIA NVFP4 description](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
and the [OCP microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
The 16/32 block widths are encoding constants, not unexplained tuning values.

## Executable paths

1. Scalar packing, decoding and bounded BF16 conversion for every listed format.
2. GPU quantization, scale generation and BF16 reconstruction for every format.
   Activation/output writes and cache publication remain explicit IR effects.
3. Compact-weight **decode-to-BF16** projections through the existing physical
   compiler pipeline, with BF16 activations and F32 accumulation. Both matrix
   and single-token dot-tree paths consume the same representation contract.
   Projection tile ownership and fragment schedules are retained, not copied
   into a separate hand-written low-bit GEMM.
4. Explicit startup expansion into BF16. The streaming adapter splits reads
   into bounded row-local windows, including odd-nibble starts and partial scale
   blocks. It does not allocate a model-sized host payload. The caller supplies
   independently authenticated readers and the final arena/upload sink.
5. A model-wide immutable weight schema assigns ordinal bindings and aligned
   arena offsets, sums every resident tensor, and accounts for sequential vs
   concurrent scratch lifetimes. Its startup effect interpreter preflights all
   placements/global scales/budgets before any read or upload and streams mixed
   formats into a single unpublished BF16 arena. It is not a replacement for
   the deployment loader's authenticated source and publication transaction.

Tile loads consume already-validated immutable payloads. Scale validation is
not repeated in the production tile loader. Conversion qualification kernels
retain explicit numerical error reporting, not authentication operations.
Reduction-stage launch geometry is returned by device lowering and consumed by
the exporter in semantic pass order. Whole-tensor conversion cannot borrow a
tile-sized scratch plan; oversized CUDA grids are rejected during AOT lowering.

## Validation on DGX Spark

GPU: GB10, `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, CUDA 13.0.
Campaigns used an 8 GiB memory limit, disabled swap, a 32 GiB available-memory
reserve, and serial GPU execution.

- Conversion: 12 formats × shapes `1×1`, `3×35`, `2×257`, 36 cases. Payload,
  scale bytes, global scale bits and reconstructed BF16 matched the independent
  MoonBit scalar implementation exactly.
- Compact projection: 12 formats × live row counts `1`, `3`, `32`, 36 cases,
  logical GEMM `M×128` by `64×128`. Outputs met the declared test tolerance;
  inactive output rows stayed untouched.
- Both campaigns passed memcheck, full leak checking, racecheck and synccheck,
  with zero errors/hazards and zero leaked allocations.
- The first two conversion exports exposed literal-rendering and fixture-name
  bugs; these were fixed. The first projection sanitizer run exposed a **probe**
  cleanup-order bug (device reset preceded four live allocation destructors);
  it was fixed and the campaign repeated in a new directory. Failed records
  were preserved, not relabeled.

Local records: `benchmarks/results/precision-20261004.J8Gma9/` (ignored artifacts).
Passing current conversion root: `qualification-launch-final`; passing projection root:
`projection-qualification2`. Downloaded executable SHA-256 values are recorded
and checked against their remote originals in each campaign.

The affected-package native suite passed 230 tests; the final full native suite
passed 4,399/4,399. Native check/test use the
repository's existing migration-warning exclusions (`-79-20-29-25-92-14`), not
new per-package suppressions. New packages use explicit trait extensions.

## Deliberately unfinished integration

This change is **not full quantized-model serving support** and does not enable
new formats in an existing signed BF16 deployment manifest.

Remaining work is concrete:

- Join versioned precision schemas and packed scale/global buffers to the
  whole-model loader, execution manifest and worker's startup bindings.
- Bind all graph projection families, including segmented QKV and gated MLP;
  qualify selected-row head execution on hardware (source tests cover it).
- Connect quantized activation/KV kernels to transactional KV write/read and
  attention execution, not merely expose their standalone conversion kernels.
- Implement native FP8/INT8/FP4 Tensor Core routes and hardware scale layouts
  where supported. The implemented compact route reconstructs BF16 tiles;
  it does **not** execute native W8A8/W4A4 MMA instructions.
- Add checkpoint-specific adapters for external packing/scales/zero points,
  such as AWQ/GPTQ. Their layouts cannot alias the canonical INT4 layout.
- Run whole-model correctness, quality and end-to-end latency/throughput tests
  before selecting these paths for serving or claiming a speed improvement.

BF16 serving defaults and the existing IR/manifest numerical identities remain
unchanged. No quantized-model throughput or production-readiness claim follows
from these kernel tests.

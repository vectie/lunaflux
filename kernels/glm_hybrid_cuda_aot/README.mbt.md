# GLM hybrid CUDA AOT candidates

This package binds GLM requirement identity and exact geometry to the shared
decoder-foundation CUDA source renderer. The initial deterministic subset is
BF16 token embedding, per-layer decoder RMSNorm, final RMSNorm, the terminal
bias-free language-model head, and the bias-free DSA attention output
projection. Both projections use the family-neutral dense renderer and bind
exact row-major `[output, input]` geometry, ordered F32 accumulation, BF16
rounding, launch dimensions, and counts-driven operand ABI. They share one
candidate contract while retaining distinct semantic families and symbols.

The family-neutral GLM requirement package intentionally does not import the
GLM-5.3 tensor-manifest or numeric-plan packages. Consequently this adapter
cannot bind their layout or numeric-plan digests without leaking family types
into the reusable GLM hybrid boundary. Official integration tests prove those
descriptors agree, but that evidence remains separate and grants no authority.

Candidates remain offline-only and non-bindable. Compilation, catalog
admission, runtime loading, launch, and numerical qualification are separate
authorities.

The MoE routing subset now includes separate F32 sigmoid/correction and serial
grouped-selection candidates. Corrected scores select groups and experts;
uncorrected sigmoid scores supply normalized and scaled selected weights. The
deterministic tie policy is explicitly not PyTorch unsorted-top-k parity.

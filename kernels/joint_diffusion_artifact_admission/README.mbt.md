# Joint diffusion artifact admission

This family-neutral package joins exact `JointDiffusionKernelRequirements` to
content-addressed module bytes, stable AOT catalog entry-point identities,
bounded CUDA symbols, exact device-target metadata, launch dimensions, and an
ordered diffusion operand ABI. Admission is immutable and digest-bound. It does
not load a module, open a device, launch a kernel, compile code, or use JIT.

Module digests are supplied labels, not checksums authenticated by admission.
No CUBIN scan occurs here; bounded sizes, unique labels, required symbols and
semantic launch compatibility remain checked. Optional payload integrity
verification belongs outside engine startup and inference.

The existing catalog and launch-contract semantic vocabularies cannot encode
this join faithfully: `CatalogEntry` and `AotLaunchContract` require
decoder-only `OperationKind`, `OperationShape`, `KernelCapabilityId`,
`OperationId`, and decoder/KV operand roles. `KernelModuleInput` also keeps its
bytes private and `artifact.admit` accepts only decoder/paged/tensor-parallel
contract sets. This package therefore reuses the faithful generic pieces
(`DeviceTarget`, content-addressed AOT family/entry-point identities,
`AotLaunchDimensions`, `KernelEntryPointInput`, and symbol validation) while
owning a separate diffusion module snapshot and operand/evidence vocabulary.
No decoder semantic is aliased to a diffusion operation.

Latent projections use separate input, projection-weight, projection-bias, and
hidden-output operands. Their dtypes and packed-row interpretation are owned
by the requirement-bound plan rather than inferred from byte counts.

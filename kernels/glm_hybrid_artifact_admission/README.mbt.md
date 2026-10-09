# GLM hybrid AOT artifact admission

This family-neutral bridge joins an exact `GlmHybridKernelRequirements` set to
content-addressed module bytes, immutable AOT entrypoint identity and function
symbol, target, launch dimensions, ordered operand ABI, and an optional
caller-declared qualification digest. It imports no GLM-5.3 family package in
production.

The current generic catalog cannot faithfully express DSA, KDA, mHC, MoE,
MTP, or vision semantics because those operations have no exact generic
`KernelCapabilityId` or `OperationShape`. This package therefore reuses only
the catalog's opaque target, module-digest, family, and entrypoint identities,
plus launch dimensions. It does not fabricate generic catalog entries.

Embedding, decoder/final RMSNorm, DSA attention-output projection, and terminal
language-model-head claims have an exact checked ABI. Step counts are five I32
cells aligned to four bytes. The DSA projection is exactly step counts, hidden
input, row-major model weights, then hidden output; the head changes only the
final semantic role to logits output. Other still-inert operations bind the
caller-supplied symbol and operand evidence into the admission digest but
remain without a runnable semantic ABI.

Admission authenticates supplied bytes but does not load them. A complete set
without a declared qualification digest is explicitly
`NonRunnableUnqualified`. Even when every caller supplies a syntactically
valid digest, the status is only
`NonRunnableWithDeclaredQualificationDigests`: these declarations are
untrusted and publicly constructible. They are not qualification authority,
device execution authority, performance qualification, or proof that a
launcher exists.

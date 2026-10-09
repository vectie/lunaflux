# AOT candidate coverage

This native, startup-only package audits structural coverage of one exact
ordered kernel-requirement set. Every candidate claim must carry the same model
identity and requirements digest, a unique in-range ordinal, and syntactically
valid source and recipe digests. The result lists missing and non-bindable
ordinals and has an input-order-independent canonical digest.

This is deliberately not an artifact catalog, compiler, qualification owner,
or execution authority. `CompleteWithBindableClaims` describes only the
caller-supplied candidate metadata; independent artifact admission, physical
qualification, device binding, and runtime construction remain mandatory.

The package imports no model family, scheduler, KV, device, CUDA, filesystem,
or service owner.

# Luna kernel capability manifest entry

This package owns the versioned typed capability entry for one specialized AOT
kernel. Its canonical signing bytes bind operation and semantic version, GPU
architecture range, dtype and accumulator, tensor/KV layout, exact execution
shape, alignment and workspace, graph metadata, module and specialization
digests, baked input identities, compiler policy, numerical policy, and four
content-addressed evidence references.

Signature verification remains at the deployment boundary defined by the
product contract. LunaFlux owns no public key store and performs no Ed25519
operation. Public `LunaExternalSignedApproval` values are inert. Only a
deployment-provisioned `LunaExternalApprovalVerifier` can authenticate the
immutable policy and exact manifest, envelope, and approved-source identities
with HMAC-SHA256 and produce an opaque `LunaAuthenticatedExternalApproval`.
There is deliberately no public arbitrary-key verifier constructor. Until a
trusted startup handoff is integrated, production promotion remains
unavailable; untrusted worker bootstrap claims fail closed.

Graph metadata declares a shape class, bounded startup memory, capture safety,
and an exact validated eager-fallback evidence digest. The only executable
disposition in this software foundation is `LunaValidatedEagerFallback`; the
type cannot claim that physical CUDA graph capture succeeded.

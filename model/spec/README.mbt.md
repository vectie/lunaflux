# Model specification contracts

This package owns validated semantic model geometry and the canonical model
identities used to salt plans, caches, and execution state. `ContentDigest`
and `PlanDigest` accept only lowercase 64-character SHA-256 text through their
checked constructors. `ModelIdentity` combines those validated digest types
without exposing their record representation and defensively revalidates both
components at combination.

The three identity records are opaque outside this package. Callers use typed
constructors and accessors rather than struct literals, so malformed digest
text cannot enter an identity by bypassing validation. Canonical plan digests
are derived from fixed schema prefixes and validated semantic model data.

`LlamaModelSpec` and `LlamaModelMetadata` are also opaque. The specification
constructor shares the complete defensive invariant checker, including
positive geometry, head relationships, the derived head dimension, numeric
policy, and supported dtypes. Metadata admission revalidates the specification
before deriving its plan identity, and the metadata invariant requires that
identity to contain the canonical plan digest for the validated specification.

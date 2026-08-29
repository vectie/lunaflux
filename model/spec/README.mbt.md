# Model specification contracts

This package owns validated semantic model geometry and checked digest
vocabulary. `ContentDigest` and `PlanDigest` accept only lowercase 64-character
SHA-256 text through their constructors. `ModelIdentity` combines those typed
claims without exposing their record representation and defensively revalidates
both components at combination; it does not authenticate plan semantics.

The three identity records are opaque outside this package. Callers use typed
constructors and accessors rather than struct literals, so malformed digest
text cannot enter an identity by bypassing validation. `PlanDigest` and
`ModelIdentity` remain checked wire/persistence vocabulary, but trusted plan
identities are minted only by a validated `model/plan.ModelPlan`.
`LlamaModelSpec` still owns the complete reference-free `ModelNumericSchema`;
the builder passes that schema into `ModelPlan`, whose canonical plan encoding
binds the schema's raw SHA-256 digest with the complete validated graph.

`LlamaModelSpec` and `LlamaModelMetadata` are also opaque. The specification
constructor shares the complete defensive invariant checker, including
positive geometry, head relationships, the derived head dimension, numeric
policy, and supported dtypes. Metadata admission revalidates the specification
and content digest before retaining them. Metadata is deliberately content-only;
only a validated `ModelPlan` can derive an execution-plan identity.

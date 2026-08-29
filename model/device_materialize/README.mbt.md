# Device-weight layout planning and materialization

`plan_layout` preserves the established dense-Llama safetensors path: it lays
out the already authenticated legacy weight bindings and records no model
numeric-schema digest. Its file inspection, allocation, streaming, and cleanup
contracts are unchanged.

`plan_numeric_layout` is a separate pure startup planner over a validated
`ModelPlan`. It walks the complete numeric tensor table in exact `TensorRef`
order, including parameter, scale, zero-point, and codebook tensors. Every
region length comes from `numeric_contract.storage_byte_length`; checked
alignment, end, materialized-sum, and arena arithmetic finish before each
region is appended. The resulting opaque layout binds the exact model identity
and numeric-schema digest that produced its geometry.

Numeric layout planning opens no file, allocates no device resource, copies no
payload, and makes no loader, artifact, execution, support, or readiness claim.
It is inert placement evidence for later numeric-aware startup stages.

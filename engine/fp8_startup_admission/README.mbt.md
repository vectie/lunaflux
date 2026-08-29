# FP8 startup admission

This package is the final pure Phase-8 join for finite-E4M3 W8A8 startup. It
cross-authenticates one exact model numeric schema, observed device numeric
capability, kernel capability manifest, launch plan, and the single admitted
artifact bundle retained by that manifest. Its canonical digest binds those
independent identities without copying module bytes or per-operation records.

The admission is intentionally inert. It is not a device context, module load,
kernel launch, physical correctness result, or readiness signal. Unsupported,
substituted, or incomplete evidence is rejected before this value can be
published. A later physical execution backend must consume and reauthenticate
this evidence and establish its own resource lifecycle and correctness gates.

`Fp8StartupAdmissionV2` is a separate staged-launch authority domain. It
requires every admitted FP8 operation to bind the plan's exact one- or
two-stage activation-scale policy and matching 4- or 8-byte Workspace. Its
canonical identity is distinct from v1, while its representation wraps the
already-validated v1 evidence rather than duplicating plan, device, or kernel
authority. Neither value grants runtime authority.

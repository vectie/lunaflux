# Kernel configuration

`config/kernels` owns the narrow immutable operator input for kernel
admission: one strict relative manifest locator, one exact lowercase SHA-256
digest, and one AOT-only admission policy. It carries no compiler, generated
source, signature-verification, filesystem-root, device, or JIT authority.

`DeploymentApprovedAotOnly` means the deployment environment has already
verified provenance and LunaFlux must still admit the digest-pinned typed
manifest and artifact contracts before readiness. `OfflineInspectionOnly`
permits metadata reporting but can never publish runtime readiness.

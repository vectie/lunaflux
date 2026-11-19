///|
pub extend KernelCatalog with @debug.Debug::{to_repr}

///|
pub extend KernelCatalog with Eq::{not_equal, equal}

///|
pub extend KernelBinding with @debug.Debug::{to_repr}

///|
pub extend KernelBinding with Eq::{not_equal, equal}

///|
pub extend AotFamilySemantics with @debug.Debug::{to_repr}

///|
pub extend AotFamilySemantics with Eq::{not_equal, equal}

///|
pub extend ResolvedKernelCatalog with @debug.Debug::{to_repr}

///|
pub extend ResolvedKernelCatalog with Eq::{not_equal, equal}

///|
pub extend CatalogEntry with @debug.Debug::{to_repr}

///|
pub extend CatalogEntry with Eq::{not_equal, equal}

///|
pub extend CatalogValidationRule with @debug.Debug::{to_repr}

///|
pub extend CatalogValidationRule with Eq::{not_equal, equal}

///|
pub extend CatalogError with @debug.Debug::{to_repr}

///|
pub extend CatalogError with Eq::{not_equal, equal}

///|
pub extend KernelExecutionSemantics with @debug.Debug::{to_repr}

///|
pub extend KernelExecutionSemantics with Eq::{not_equal, equal}

///|
pub extend PagedKvLayoutContract with @debug.Debug::{to_repr}

///|
pub extend PagedKvLayoutContract with Eq::{not_equal, equal}

///|
pub extend CatalogVersion with @debug.Debug::{to_repr}

///|
pub extend CatalogVersion with Eq::{not_equal, equal}

///|
pub extend KernelSemanticVersion with @debug.Debug::{to_repr}

///|
pub extend KernelSemanticVersion with Eq::{not_equal, equal}

///|
pub extend AotArtifactDigest with @debug.Debug::{to_repr}

///|
pub extend AotArtifactDigest with Eq::{not_equal, equal}

///|
pub extend AotArtifactDigest with Hash::{hash, hash_combine}

///|
pub extend AotKernelFamilyId with @debug.Debug::{to_repr}

///|
pub extend AotKernelFamilyId with Eq::{not_equal, equal}

///|
pub extend AotKernelFamilyId with Hash::{hash, hash_combine}

///|
pub extend AotKernelFamily with @debug.Debug::{to_repr}

///|
pub extend AotKernelFamily with Eq::{not_equal, equal}

///|
pub extend AotKernelFamily with Hash::{hash, hash_combine}

///|
pub extend AotKernelEntryPointId with @debug.Debug::{to_repr}

///|
pub extend AotKernelEntryPointId with Eq::{not_equal, equal}

///|
pub extend AotKernelEntryPointId with Hash::{hash, hash_combine}

///|
pub extend AotKernelEntryPoint with @debug.Debug::{to_repr}

///|
pub extend AotKernelEntryPoint with Eq::{not_equal, equal}

///|
pub extend AotKernelEntryPoint with Hash::{hash, hash_combine}

///|
pub extend KernelImplementationId with @debug.Debug::{to_repr}

///|
pub extend KernelImplementationId with Eq::{not_equal, equal}

///|
pub extend DeviceTarget with @debug.Debug::{to_repr}

///|
pub extend DeviceTarget with Eq::{not_equal, equal}

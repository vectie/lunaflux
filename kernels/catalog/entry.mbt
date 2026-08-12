///|
/// One immutable exact-match catalog entry.
pub struct CatalogEntry {
  semantic_version : KernelSemanticVersion
  operation_kind : @plan.OperationKind
  operation_shape : @plan.OperationShape
  capability : @plan.KernelCapabilityId
  compute_major : Int
  compute_minor : Int
  implementation : KernelImplementationId
  workspace_bytes : Int64
  workspace_alignment : Int64
} derive(Debug, Eq)

///|
fn is_power_of_two(value : Int64) -> Bool {
  if value <= 0L {
    return false
  }
  for remaining = value; remaining % 2L == 0L; {
    continue remaining / 2L
  } nobreak {
    remaining == 1L
  }
}

///|
fn vendor_implementation_is_legal(
  operation_kind : @plan.OperationKind,
  implementation : KernelImplementationId,
) -> Bool {
  match implementation {
    VendorBf16GemmImplicitHeuristicV1 =>
      operation_kind is (OutputProjection | LanguageModelHead)
    ContentAddressedAot(_) => true
  }
}

///|
fn operation_shape_is_legal(
  operation_kind : @plan.OperationKind,
  operation_shape : @plan.OperationShape,
) -> Bool {
  match (operation_kind, operation_shape) {
    (Embedding, EmbeddingShape(_)) => true
    (RmsNorm, NormalizationShape(_)) => true
    (QkvProjection | OutputProjection | LanguageModelHead, ProjectionShape(_)) =>
      true
    (RotaryEmbedding, RotaryShape(_)) => true
    (CausalAttention, AttentionShape(_)) => true
    (ResidualAdd, ElementwiseShape(_)) => true
    (GatedMlp, GatedMlpShape(_)) => true
    _ => false
  }
}

///|
fn operation_capability_is_legal(
  operation_kind : @plan.OperationKind,
  capability : @plan.KernelCapabilityId,
) -> Bool {
  let expected = match operation_kind {
    Embedding => @plan.KernelCapabilityId::embedding_lookup()
    RmsNorm => @plan.KernelCapabilityId::rms_norm()
    QkvProjection => @plan.KernelCapabilityId::qkv_projection()
    RotaryEmbedding => @plan.KernelCapabilityId::rotary_embedding()
    CausalAttention => @plan.KernelCapabilityId::causal_attention()
    OutputProjection | LanguageModelHead =>
      @plan.KernelCapabilityId::dense_projection()
    ResidualAdd => @plan.KernelCapabilityId::residual_add()
    GatedMlp => @plan.KernelCapabilityId::gated_mlp()
  }
  capability == expected
}

///|
pub fn CatalogEntry::new(
  semantic_version~ : KernelSemanticVersion,
  operation_kind~ : @plan.OperationKind,
  operation_shape~ : @plan.OperationShape,
  capability~ : @plan.KernelCapabilityId,
  compute_major~ : Int,
  compute_minor~ : Int,
  implementation~ : KernelImplementationId,
  workspace_bytes~ : Int64,
  workspace_alignment~ : Int64,
) -> CatalogEntry raise CatalogError {
  if semantic_version != KernelSemanticVersion::v1() {
    raise InvalidCatalog(UnsupportedSemanticVersion)
  }
  if compute_major <= 0 || compute_minor < 0 {
    raise InvalidCatalog(InvalidComputeCapability)
  }
  if workspace_bytes < 0L {
    raise InvalidCatalog(InvalidWorkspaceBytes)
  }
  if !is_power_of_two(workspace_alignment) || workspace_alignment > 65536L {
    raise InvalidCatalog(InvalidWorkspaceAlignment)
  }
  if !operation_shape_is_legal(operation_kind, operation_shape) {
    raise InvalidCatalog(OperationShapeMismatch)
  }
  if !operation_capability_is_legal(operation_kind, capability) {
    raise InvalidCatalog(OperationCapabilityMismatch)
  }
  if !vendor_implementation_is_legal(operation_kind, implementation) {
    raise InvalidCatalog(InvalidVendorImplementationKind)
  }
  {
    semantic_version,
    operation_kind,
    operation_shape,
    capability,
    compute_major,
    compute_minor,
    implementation,
    workspace_bytes,
    workspace_alignment,
  }
}

///|
pub fn CatalogEntry::semantic_version(
  self : CatalogEntry,
) -> KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn CatalogEntry::operation_kind(self : CatalogEntry) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn CatalogEntry::operation_shape(
  self : CatalogEntry,
) -> @plan.OperationShape {
  self.operation_shape
}

///|
pub fn CatalogEntry::capability(
  self : CatalogEntry,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn CatalogEntry::compute_major(self : CatalogEntry) -> Int {
  self.compute_major
}

///|
pub fn CatalogEntry::compute_minor(self : CatalogEntry) -> Int {
  self.compute_minor
}

///|
pub fn CatalogEntry::implementation(
  self : CatalogEntry,
) -> KernelImplementationId {
  self.implementation
}

///|
pub fn CatalogEntry::workspace_bytes(self : CatalogEntry) -> Int64 {
  self.workspace_bytes
}

///|
pub fn CatalogEntry::workspace_alignment(self : CatalogEntry) -> Int64 {
  self.workspace_alignment
}

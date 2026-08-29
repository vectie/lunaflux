///|
pub struct LunaGraphShapeClass {
  priv phase : @device_profile.ExecutionPhase
  priv batch_rows : Int
  priv query_tokens_per_row : Int
  priv context_tokens : Int
  priv token_rows : Int64
} derive(Debug, Eq)

///|
pub fn LunaGraphShapeClass::from_execution_shape(
  shape : @device_profile.ExactExecutionShape,
) -> LunaGraphShapeClass {
  {
    phase: shape.phase(),
    batch_rows: shape.batch_rows(),
    query_tokens_per_row: shape.query_tokens_per_row(),
    context_tokens: shape.context_tokens(),
    token_rows: shape.token_rows(),
  }
}

///|
pub fn LunaGraphShapeClass::batch_rows(self : LunaGraphShapeClass) -> Int {
  self.batch_rows
}

///|
pub fn LunaGraphShapeClass::phase(
  self : LunaGraphShapeClass,
) -> @device_profile.ExecutionPhase {
  self.phase
}

///|
pub fn LunaGraphShapeClass::query_tokens_per_row(
  self : LunaGraphShapeClass,
) -> Int {
  self.query_tokens_per_row
}

///|
pub fn LunaGraphShapeClass::context_tokens(self : LunaGraphShapeClass) -> Int {
  self.context_tokens
}

///|
pub fn LunaGraphShapeClass::token_rows(self : LunaGraphShapeClass) -> Int64 {
  self.token_rows
}

///|
pub struct LunaValidatedEagerFallbackDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaValidatedEagerFallbackDigest::from_sha256(
  value : String,
) -> LunaValidatedEagerFallbackDigest raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(value) {
    raise Invalid(EagerFallback)
  }
  LunaValidatedEagerFallbackDigest(value)
}

///|
pub fn LunaValidatedEagerFallbackDigest::as_hex(
  self : LunaValidatedEagerFallbackDigest,
) -> String {
  self.0
}

///|
pub struct LunaGraphCapabilityMetadata {
  priv shape_class : LunaGraphShapeClass
  priv capture_safe : Bool
  priv graph_memory_bytes : Int64
  priv graph_memory_alignment : Int64
  priv eager_fallback : LunaValidatedEagerFallbackDigest
} derive(Debug, Eq)

///|
pub fn LunaGraphCapabilityMetadata::new(
  shape_class~ : LunaGraphShapeClass,
  capture_safe~ : Bool,
  graph_memory_bytes~ : Int64,
  graph_memory_alignment~ : Int64,
  eager_fallback~ : LunaValidatedEagerFallbackDigest,
) -> LunaGraphCapabilityMetadata raise LunaCapabilityManifestError {
  if graph_memory_bytes < 0L ||
    graph_memory_alignment <= 0L ||
    (graph_memory_alignment & (graph_memory_alignment - 1L)) != 0L ||
    graph_memory_bytes % graph_memory_alignment != 0L ||
    (!capture_safe && graph_memory_bytes != 0L) ||
    (capture_safe && graph_memory_bytes == 0L) {
    raise Invalid(GraphMemory)
  }
  {
    shape_class,
    capture_safe,
    graph_memory_bytes,
    graph_memory_alignment,
    eager_fallback,
  }
}

///|
pub fn LunaGraphCapabilityMetadata::shape_class(
  self : LunaGraphCapabilityMetadata,
) -> LunaGraphShapeClass {
  self.shape_class
}

///|
pub fn LunaGraphCapabilityMetadata::capture_safe(
  self : LunaGraphCapabilityMetadata,
) -> Bool {
  self.capture_safe
}

///|
pub fn LunaGraphCapabilityMetadata::graph_memory_bytes(
  self : LunaGraphCapabilityMetadata,
) -> Int64 {
  self.graph_memory_bytes
}

///|
pub fn LunaGraphCapabilityMetadata::graph_memory_alignment(
  self : LunaGraphCapabilityMetadata,
) -> Int64 {
  self.graph_memory_alignment
}

///|
pub fn LunaGraphCapabilityMetadata::eager_fallback_digest(
  self : LunaGraphCapabilityMetadata,
) -> LunaValidatedEagerFallbackDigest {
  self.eager_fallback
}

///|
pub(all) enum LunaGraphExecutionDisposition {
  LunaValidatedEagerFallback
} derive(Debug, Eq)

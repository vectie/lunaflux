///|
pub fn GlmHybridKernelRequirement::operation(
  self : Self,
) -> GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKernelRequirement::capability(
  self : Self,
) -> GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKernelRequirementsDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn GlmHybridKernelRequirements::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKernelRequirements::source_schema_version(self : Self) -> Int {
  self.source_schema_version
}

///|
pub fn GlmHybridKernelRequirements::geometry(
  self : Self,
) -> GlmHybridKernelGeometry {
  self.geometry
}

///|
pub fn GlmHybridKernelGeometry::vocabulary_size(self : Self) -> Int {
  self.vocabulary_size
}

///|
pub fn GlmHybridKernelGeometry::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn GlmHybridKernelGeometry::dense_intermediate_size(self : Self) -> Int {
  self.dense_intermediate_size
}

///|
pub fn GlmHybridKernelGeometry::decoder_layer_count(self : Self) -> Int {
  self.decoder_layer_count
}

///|
pub fn GlmHybridKernelGeometry::context_length(self : Self) -> Int {
  self.context_length
}

///|
pub fn GlmHybridKernelRequirements::digest(
  self : Self,
) -> GlmHybridKernelRequirementsDigest {
  self.digest
}

///|
pub fn GlmHybridKernelRequirements::requirements(
  self : Self,
) -> ReadOnlyArray[GlmHybridKernelRequirement] {
  self.requirements
}

///|
pub fn GlmHybridKernelRequirements::capabilities(
  self : Self,
) -> ReadOnlyArray[GlmHybridAotCapability] {
  self.capabilities
}

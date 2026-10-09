///|
fn ModelSemanticCapability::stable_index(self : Self) -> Int {
  match self {
    DenseCausalAttention => 0
    QueryKeyRmsNorm => 1
    MixtureOfExperts => 2
    MultiTokenPrediction => 3
    HyperConnections => 4
    CompressedSparseAttention => 5
    HeavilyCompressedAttention => 6
    DynamicSparseAttention => 7
    LinearRecurrentAttention => 8
    VisionEncoder => 9
    MultimodalProjection => 10
    DiffusionDenoising => 11
    VideoVae => 12
    AudioVae => 13
    ReferenceConditioning => 14
    TextConditioningEncoder => 15
  }
}

///|
fn capability_count() -> Int {
  16
}

///|
pub fn ModelBackendCapabilities::new(
  capabilities : ArrayView[ModelSemanticCapability],
) -> ModelBackendCapabilities raise FamilyContractError {
  let supported = FixedArray::make(capability_count(), false)
  for capability in capabilities {
    let index = capability.stable_index()
    if supported[index] {
      raise DuplicateCapability(capability)
    }
    supported[index] = true
  }
  { supported, }
}

///|
pub fn ModelBackendCapabilities::supports(
  self : Self,
  capability : ModelSemanticCapability,
) -> Bool {
  self.supported[capability.stable_index()]
}

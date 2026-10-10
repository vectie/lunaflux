///|
pub fn MiniMaxH3SemanticTensorBinding::source_index(self : Self) -> Int {
  self.source_index
}

///|
pub fn MiniMaxH3SemanticTensorBinding::role(self : Self) -> MiniMaxH3TensorRole {
  self.role
}

///|
pub fn MiniMaxH3SemanticTensorBinding::tensor(
  self : Self,
) -> @safetensors.TensorMetadata {
  self.tensor
}

///|
pub fn MiniMaxH3PartialShardBinding::component(
  self : Self,
) -> MiniMaxH3WeightComponent {
  self.component
}

///|
pub fn MiniMaxH3PartialShardBinding::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn MiniMaxH3PartialShardBinding::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn MiniMaxH3PartialShardBinding::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn MiniMaxH3PartialShardBinding::tensor(
  self : Self,
  index : Int,
) -> MiniMaxH3SemanticTensorBinding? {
  self.tensors.get(index)
}

///|
pub fn MiniMaxH3CompleteComponentBinding::component(
  self : Self,
) -> MiniMaxH3WeightComponent {
  self.component
}

///|
pub fn MiniMaxH3CompleteComponentBinding::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn MiniMaxH3CompleteComponentBinding::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn MiniMaxH3CompleteComponentBinding::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn MiniMaxH3CompleteComponentBinding::tensor(
  self : Self,
  index : Int,
) -> MiniMaxH3SemanticTensorBinding? {
  self.tensors.get(index)
}

///|
pub fn MiniMaxH3CompleteComponentBinding::require_materialization(
  self : Self,
) -> Unit raise MiniMaxH3WeightsError {
  raise MaterializationUnavailable(self.component)
}

///|
pub fn MiniMaxH3WeightComponent::conditioning_uses(
  self : Self,
) -> ReadOnlyArray[MiniMaxH3ConditioningUse] {
  match self {
    Fl2vaDenoiser => [Fl2vaKeyframeConditioning]
    Ref2vaDenoiser =>
      [
        Ref2vaImageConditioning,
        Ref2vaVideoConditioning,
        Ref2vaAudioConditioning,
      ]
    TextConditioner =>
      [
        PromptConditioning,
        Fl2vaKeyframeConditioning,
        Ref2vaImageConditioning,
        Ref2vaVideoConditioning,
      ]
    VideoVae => [Fl2vaKeyframeConditioning, Ref2vaVideoConditioning]
    AudioVae => [Ref2vaAudioConditioning]
  }
}

///|
/// Converted schema-v1 source tensor count, also an upper bound for the
/// original VideoVAE layout. Its exact 560-name vocabulary is owned by
/// MiniMaxH3OriginalVideoVaeSchema; complete bindings expose their actual count.
pub fn MiniMaxH3WeightComponent::required_tensor_count(self : Self) -> Int {
  complete_vocabulary(self).length()
}

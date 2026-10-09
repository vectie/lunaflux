///|
fn new_workload(
  kind : WorkloadExecutionKind,
  identity : @spec.ModelIdentity,
  schema_version : Int,
  decoder_layer_count : Int?,
  maximum_token_context : Int?,
) -> WorkloadExecutionPlan raise WorkloadExecutionPlanError {
  if schema_version <= 0 {
    raise InvalidSchemaVersion
  }
  match (kind, decoder_layer_count, maximum_token_context) {
    (
      DecoderOnlyText
      | MultimodalConditionalGeneration,
      Some(layers),
      Some(context),
    ) => if layers <= 0 || context <= 0 { raise InvalidDecoderContract }
    (JointAudioVideoDiffusion, None, None) => ()
    (DecoderOnlyText | MultimodalConditionalGeneration, _, _) =>
      raise InvalidDecoderContract
    (JointAudioVideoDiffusion, _, _) => raise InvalidDiffusionContract
  }
  {
    kind,
    identity,
    schema_version,
    decoder_layer_count,
    maximum_token_context,
  }
}

///|
pub fn WorkloadExecutionPlan::decoder_only(
  identity~ : @spec.ModelIdentity,
  schema_version~ : Int,
  decoder_layer_count~ : Int,
  maximum_token_context~ : Int,
) -> WorkloadExecutionPlan raise WorkloadExecutionPlanError {
  new_workload(
    DecoderOnlyText,
    identity,
    schema_version,
    Some(decoder_layer_count),
    Some(maximum_token_context),
  )
}

///|
pub fn WorkloadExecutionPlan::multimodal_conditional(
  identity~ : @spec.ModelIdentity,
  schema_version~ : Int,
  decoder_layer_count~ : Int,
  maximum_token_context~ : Int,
) -> WorkloadExecutionPlan raise WorkloadExecutionPlanError {
  new_workload(
    MultimodalConditionalGeneration,
    identity,
    schema_version,
    Some(decoder_layer_count),
    Some(maximum_token_context),
  )
}

///|
pub fn WorkloadExecutionPlan::joint_audio_video_diffusion(
  identity~ : @spec.ModelIdentity,
  schema_version~ : Int,
) -> WorkloadExecutionPlan raise WorkloadExecutionPlanError {
  new_workload(JointAudioVideoDiffusion, identity, schema_version, None, None)
}

///|
pub fn WorkloadExecutionPlan::kind(self : Self) -> WorkloadExecutionKind {
  self.kind
}

///|
pub fn WorkloadExecutionPlan::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn WorkloadExecutionPlan::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn WorkloadExecutionPlan::decoder_layer_count(self : Self) -> Int? {
  self.decoder_layer_count
}

///|
pub fn WorkloadExecutionPlan::maximum_token_context(self : Self) -> Int? {
  self.maximum_token_context
}

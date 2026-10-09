///|
/// Released MiniMax H3 base checkpoint family.
pub(all) enum MiniMaxH3Variant {
  Fl2va
  Ref2va
} derive(Debug, Eq)

///|
/// Official checkpoint metadata layout normalized by this package.
///
/// `OriginalV1` names the `MiniMaxH3DiTModel` configuration shipped below
/// `FL2VA/` and `Ref2VA/`. `DiffusersV1` names the equivalent
/// `MiniMaxH3Transformer3DModel` configuration at the repository root.
pub(all) enum MiniMaxH3CheckpointFormat {
  OriginalV1
  DiffusersV1
} derive(Debug, Eq)

///|
/// Official checkpoint storage profile. Input/output projections and the
/// timestep MLP are F32; the context path and transformer stack are BF16.
pub(all) enum MiniMaxH3PrecisionProfile {
  OfficialF32BoundaryBFloat16Core
} derive(Debug, Eq)

///|
/// Exact official configuration field rejected by schema-v1 admission.
pub(all) enum MiniMaxH3ConfigField {
  HiddenSize
  LayerCount
  RefinerLayerCount
  AttentionHeadCount
  AttentionHeadDimension
  FfnDimension
  VisualLatentChannels
  AudioLatentChannels
  PatchTemporal
  PatchHeight
  PatchWidth
  TextDimension
  FrequencyDimension
  TimeEmbeddingHiddenDimension
  TimeEmbeddingDimension
  AdaLnOutputFeatures
  FinalAdaLnOutputFeatures
  RopeFrequencyDimension
  RopeTheta
  NormEpsilon
  QueryKeyNormEpsilon
  FinalNormEpsilon
} derive(Debug, Eq)

///|
/// Payload-safe MiniMax H3 configuration admission failures.
pub(all) suberror MiniMaxH3SpecError {
  UnsupportedSchemaVersion
  InvalidOfficialValue(MiniMaxH3ConfigField)
  UnsupportedWeightDType
} derive(Debug, Eq)

///|
/// Exact normalized schema-v1 profile for the released MiniMax H3 base
/// Omni Transformer.
///
/// This type validates metadata only. It does not claim that LunaFlux's
/// decoder-only `ModelPlan` can represent audio/video diffusion execution.
pub struct MiniMaxH3TransformerSpec {
  priv variant : MiniMaxH3Variant
  priv checkpoint_format : MiniMaxH3CheckpointFormat
  priv schema_version : Int
  priv hidden_size : Int
  priv layer_count : Int
  priv refiner_layer_count : Int
  priv attention_head_count : Int
  priv attention_head_dimension : Int
  priv ffn_dimension : Int
  priv visual_latent_channels : Int
  priv audio_latent_channels : Int
  priv patch_temporal : Int
  priv patch_height : Int
  priv patch_width : Int
  priv text_dimension : Int
  priv frequency_dimension : Int
  priv time_embedding_hidden_dimension : Int
  priv time_embedding_dimension : Int
  priv adaln_output_features : Int
  priv final_adaln_output_features : Int
  priv rope_frequency_dimension : Int
  priv rope_theta : Double
  priv norm_epsilon : Double
  priv query_key_norm_epsilon : Double
  priv final_norm_epsilon : Double
  priv core_weight_dtype : @spec.ModelDType
  priv precision_profile : MiniMaxH3PrecisionProfile
} derive(Debug, Eq)

///|
/// Immutable MiniMax H3 profile bound to independently verified model
/// content. The digest is content identity, not execution-plan authority.
pub struct MiniMaxH3ModelMetadata {
  priv spec : MiniMaxH3TransformerSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
/// Exact `_class_name` accepted for this checkpoint metadata layout.
pub fn MiniMaxH3CheckpointFormat::architecture(self : Self) -> String {
  match self {
    OriginalV1 => "MiniMaxH3DiTModel"
    DiffusersV1 => "MiniMaxH3Transformer3DModel"
  }
}

///|
fn require_exact_int(
  actual : Int,
  expected : Int,
  field : MiniMaxH3ConfigField,
) -> Unit raise MiniMaxH3SpecError {
  if actual != expected {
    raise InvalidOfficialValue(field)
  }
}

///|
fn require_exact_double(
  actual : Double,
  expected : Double,
  field : MiniMaxH3ConfigField,
) -> Unit raise MiniMaxH3SpecError {
  if actual.is_nan() || actual.is_inf() || actual != expected {
    raise InvalidOfficialValue(field)
  }
}

///|
/// Validates the exact normalized configuration published for the official
/// MiniMax H3 base FL2VA and Ref2VA checkpoints.
///
/// The two official layouts use different JSON field names; callers normalize
/// those names before this boundary. No nearby geometry or precision is
/// silently accepted as MiniMax H3 schema v1.
pub fn MiniMaxH3TransformerSpec::new_exact(
  variant~ : MiniMaxH3Variant,
  checkpoint_format~ : MiniMaxH3CheckpointFormat,
  schema_version~ : Int,
  hidden_size~ : Int,
  layer_count~ : Int,
  refiner_layer_count~ : Int,
  attention_head_count~ : Int,
  attention_head_dimension~ : Int,
  ffn_dimension~ : Int,
  visual_latent_channels~ : Int,
  audio_latent_channels~ : Int,
  patch_temporal~ : Int,
  patch_height~ : Int,
  patch_width~ : Int,
  text_dimension~ : Int,
  frequency_dimension~ : Int,
  time_embedding_hidden_dimension~ : Int,
  time_embedding_dimension~ : Int,
  adaln_output_features~ : Int,
  final_adaln_output_features~ : Int,
  rope_frequency_dimension~ : Int,
  rope_theta~ : Double,
  norm_epsilon~ : Double,
  query_key_norm_epsilon~ : Double,
  final_norm_epsilon~ : Double,
  weight_dtype~ : @spec.ModelDType,
) -> MiniMaxH3TransformerSpec raise MiniMaxH3SpecError {
  if schema_version != 1 {
    raise UnsupportedSchemaVersion
  }
  require_exact_int(hidden_size, 5376, HiddenSize)
  require_exact_int(layer_count, 50, LayerCount)
  require_exact_int(refiner_layer_count, 2, RefinerLayerCount)
  require_exact_int(attention_head_count, 56, AttentionHeadCount)
  require_exact_int(attention_head_dimension, 128, AttentionHeadDimension)
  require_exact_int(ffn_dimension, 14336, FfnDimension)
  require_exact_int(visual_latent_channels, 24, VisualLatentChannels)
  require_exact_int(audio_latent_channels, 32, AudioLatentChannels)
  require_exact_int(patch_temporal, 1, PatchTemporal)
  require_exact_int(patch_height, 2, PatchHeight)
  require_exact_int(patch_width, 2, PatchWidth)
  require_exact_int(text_dimension, 5120, TextDimension)
  require_exact_int(frequency_dimension, 256, FrequencyDimension)
  require_exact_int(
    time_embedding_hidden_dimension,
    5376,
    TimeEmbeddingHiddenDimension,
  )
  require_exact_int(time_embedding_dimension, 2688, TimeEmbeddingDimension)
  require_exact_int(adaln_output_features, 96768, AdaLnOutputFeatures)
  require_exact_int(
    final_adaln_output_features,
    10752,
    FinalAdaLnOutputFeatures,
  )
  require_exact_int(rope_frequency_dimension, 16, RopeFrequencyDimension)
  require_exact_double(rope_theta, 10000.0, RopeTheta)
  require_exact_double(norm_epsilon, 0.00001, NormEpsilon)
  require_exact_double(query_key_norm_epsilon, 0.00001, QueryKeyNormEpsilon)
  require_exact_double(final_norm_epsilon, 0.00001, FinalNormEpsilon)
  if weight_dtype != BFloat16 {
    raise UnsupportedWeightDType
  }
  {
    variant,
    checkpoint_format,
    schema_version,
    hidden_size,
    layer_count,
    refiner_layer_count,
    attention_head_count,
    attention_head_dimension,
    ffn_dimension,
    visual_latent_channels,
    audio_latent_channels,
    patch_temporal,
    patch_height,
    patch_width,
    text_dimension,
    frequency_dimension,
    time_embedding_hidden_dimension,
    time_embedding_dimension,
    adaln_output_features,
    final_adaln_output_features,
    rope_frequency_dimension,
    rope_theta,
    norm_epsilon,
    query_key_norm_epsilon,
    final_norm_epsilon,
    core_weight_dtype: weight_dtype,
    precision_profile: OfficialF32BoundaryBFloat16Core,
  }
}

///|
/// Constructs one of the exact official schema-v1 profiles.
pub fn MiniMaxH3TransformerSpec::official(
  variant~ : MiniMaxH3Variant,
  checkpoint_format~ : MiniMaxH3CheckpointFormat,
) -> MiniMaxH3TransformerSpec {
  try! MiniMaxH3TransformerSpec::new_exact(
    variant~,
    checkpoint_format~,
    schema_version=1,
    hidden_size=5376,
    layer_count=50,
    refiner_layer_count=2,
    attention_head_count=56,
    attention_head_dimension=128,
    ffn_dimension=14336,
    visual_latent_channels=24,
    audio_latent_channels=32,
    patch_temporal=1,
    patch_height=2,
    patch_width=2,
    text_dimension=5120,
    frequency_dimension=256,
    time_embedding_hidden_dimension=5376,
    time_embedding_dimension=2688,
    adaln_output_features=96768,
    final_adaln_output_features=10752,
    rope_frequency_dimension=16,
    rope_theta=10000.0,
    norm_epsilon=0.00001,
    query_key_norm_epsilon=0.00001,
    final_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
  )
}

///|
pub fn MiniMaxH3TransformerSpec::variant(self : Self) -> MiniMaxH3Variant {
  self.variant
}

///|
pub fn MiniMaxH3TransformerSpec::checkpoint_format(
  self : Self,
) -> MiniMaxH3CheckpointFormat {
  self.checkpoint_format
}

///|
pub fn MiniMaxH3TransformerSpec::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn MiniMaxH3TransformerSpec::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn MiniMaxH3TransformerSpec::layer_count(self : Self) -> Int {
  self.layer_count
}

///|
pub fn MiniMaxH3TransformerSpec::refiner_layer_count(self : Self) -> Int {
  self.refiner_layer_count
}

///|
pub fn MiniMaxH3TransformerSpec::attention_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
pub fn MiniMaxH3TransformerSpec::attention_head_dimension(self : Self) -> Int {
  self.attention_head_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::attention_width(self : Self) -> Int {
  self.attention_head_count * self.attention_head_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::ffn_dimension(self : Self) -> Int {
  self.ffn_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::visual_latent_channels(self : Self) -> Int {
  self.visual_latent_channels
}

///|
pub fn MiniMaxH3TransformerSpec::audio_latent_channels(self : Self) -> Int {
  self.audio_latent_channels
}

///|
pub fn MiniMaxH3TransformerSpec::patch_shape(self : Self) -> ReadOnlyArray[Int] {
  [self.patch_temporal, self.patch_height, self.patch_width]
}

///|
pub fn MiniMaxH3TransformerSpec::text_dimension(self : Self) -> Int {
  self.text_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::frequency_dimension(self : Self) -> Int {
  self.frequency_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::time_embedding_hidden_dimension(
  self : Self,
) -> Int {
  self.time_embedding_hidden_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::time_embedding_dimension(self : Self) -> Int {
  self.time_embedding_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::adaln_output_features(self : Self) -> Int {
  self.adaln_output_features
}

///|
pub fn MiniMaxH3TransformerSpec::final_adaln_output_features(
  self : Self,
) -> Int {
  self.final_adaln_output_features
}

///|
pub fn MiniMaxH3TransformerSpec::rope_frequency_dimension(self : Self) -> Int {
  self.rope_frequency_dimension
}

///|
pub fn MiniMaxH3TransformerSpec::rope_theta(self : Self) -> Double {
  self.rope_theta
}

///|
pub fn MiniMaxH3TransformerSpec::norm_epsilon(self : Self) -> Double {
  self.norm_epsilon
}

///|
pub fn MiniMaxH3TransformerSpec::query_key_norm_epsilon(self : Self) -> Double {
  self.query_key_norm_epsilon
}

///|
pub fn MiniMaxH3TransformerSpec::final_norm_epsilon(self : Self) -> Double {
  self.final_norm_epsilon
}

///|
pub fn MiniMaxH3TransformerSpec::weight_dtype(self : Self) -> @spec.ModelDType {
  self.core_weight_dtype
}

///|
pub fn MiniMaxH3TransformerSpec::precision_profile(
  self : Self,
) -> MiniMaxH3PrecisionProfile {
  self.precision_profile
}

///|
pub fn MiniMaxH3TransformerSpec::boundary_weight_dtype(
  _self : Self,
) -> @spec.ModelDType {
  Float32
}

///|
pub fn MiniMaxH3ModelMetadata::from_verified_content(
  spec : MiniMaxH3TransformerSpec,
  content : @spec.ContentDigest,
) -> MiniMaxH3ModelMetadata {
  { spec, content_digest: content, }
}

///|
pub fn MiniMaxH3ModelMetadata::spec(self : Self) -> MiniMaxH3TransformerSpec {
  self.spec
}

///|
pub fn MiniMaxH3ModelMetadata::content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.content_digest
}

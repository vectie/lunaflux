///|
pub(all) enum MiniMaxH3ConfigDocument {
  PipelineIndex
  Transformer
  VideoVae
  AudioVae
  VideoScheduler
  AudioScheduler
} derive(Debug, Eq)

///|
pub(all) suberror MiniMaxH3ConfigError {
  InvalidLimit
  ByteLimitExceeded
  InvalidUtf8
  NestingLimitExceeded
  DuplicateField
  MalformedJson
  NonObjectRoot
  UnknownField(index~ : Int)
  MissingField
  InvalidField
  UnsupportedExecutableMetadata
  WrongDocument(MiniMaxH3ConfigDocument)
} derive(Debug, Eq)

///|
pub struct MiniMaxH3PipelineIndex {
  priv variant : @minimax_h3_spec.MiniMaxH3Variant
  priv schema_version : Int
  priv video_shift : Double
  priv audio_shift : Double
} derive(Debug, Eq)

///|
pub struct MiniMaxH3VideoVaeConfig {
  priv architecture_schema_version : Int
  priv input_channels : Int
  priv output_channels : Int
  priv latent_channels : Int
  priv block_out_channels : ReadOnlyArray[Int]
  priv layers_per_block : Int
  priv spatial_downsample_factors : ReadOnlyArray[Int]
  priv temporal_downsample_factors : ReadOnlyArray[Int]
  priv norm_group_count : Int
  priv norm_epsilon : Double
  priv clip_length : Int
  priv token_drop : Int
  priv spatial_compression : Int
  priv temporal_compression : Int
  priv decoder_layer_count : Int
  priv decoder_attention_head_count : Int
  priv decoder_attention_head_dimension : Int
  priv decoder_register_token_count : Int
  priv decoder_ffn_multiplier : Int
  priv decoder_rope_theta : Double
  priv decoder_rope_dimension_ratio : Double
  priv decoder_norm_epsilon : Double
  priv latent_means : ReadOnlyArray[Float]
  priv latent_standard_deviations : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub struct MiniMaxH3AudioVaeConfig {
  priv architecture_schema_version : Int
  priv encoder_dimension : Int
  priv encoder_rates : ReadOnlyArray[Int]
  priv latent_dimension : Int
  priv latent_channels : Int
  priv decoder_dimension : Int
  priv decoder_rates : ReadOnlyArray[Int]
  priv decoder_kernel_sizes : ReadOnlyArray[Int]
  priv attention_head_count : Int
  priv resblock_kernel_sizes : ReadOnlyArray[Int]
  priv resblock_dilation_sizes : ReadOnlyArray[ReadOnlyArray[Int]]
  priv sampling_rate : Int
  priv output_channels : Int
  priv latent_rate : Int
  priv latent_means : ReadOnlyArray[Float]
  priv latent_standard_deviations : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub struct MiniMaxH3SchedulerConfig {
  priv document : MiniMaxH3ConfigDocument
  priv shift : Double
} derive(Debug, Eq)

///|
pub fn MiniMaxH3PipelineIndex::variant(
  self : Self,
) -> @minimax_h3_spec.MiniMaxH3Variant {
  self.variant
}

///|
pub fn MiniMaxH3PipelineIndex::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn MiniMaxH3PipelineIndex::video_shift(self : Self) -> Double {
  self.video_shift
}

///|
pub fn MiniMaxH3PipelineIndex::audio_shift(self : Self) -> Double {
  self.audio_shift
}

///|
pub fn MiniMaxH3VideoVaeConfig::latent_channels(self : Self) -> Int {
  self.latent_channels
}

///|
pub fn MiniMaxH3VideoVaeConfig::architecture_schema_version(self : Self) -> Int {
  self.architecture_schema_version
}

///|
pub fn MiniMaxH3VideoVaeConfig::input_channels(self : Self) -> Int {
  self.input_channels
}

///|
pub fn MiniMaxH3VideoVaeConfig::output_channels(self : Self) -> Int {
  self.output_channels
}

///|
pub fn MiniMaxH3VideoVaeConfig::block_out_channels(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.block_out_channels
}

///|
pub fn MiniMaxH3VideoVaeConfig::layers_per_block(self : Self) -> Int {
  self.layers_per_block
}

///|
pub fn MiniMaxH3VideoVaeConfig::spatial_downsample_factors(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.spatial_downsample_factors
}

///|
pub fn MiniMaxH3VideoVaeConfig::temporal_downsample_factors(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.temporal_downsample_factors
}

///|
pub fn MiniMaxH3VideoVaeConfig::norm_group_count(self : Self) -> Int {
  self.norm_group_count
}

///|
pub fn MiniMaxH3VideoVaeConfig::norm_epsilon(self : Self) -> Double {
  self.norm_epsilon
}

///|
pub fn MiniMaxH3VideoVaeConfig::clip_length(self : Self) -> Int {
  self.clip_length
}

///|
pub fn MiniMaxH3VideoVaeConfig::token_drop(self : Self) -> Int {
  self.token_drop
}

///|
pub fn MiniMaxH3VideoVaeConfig::spatial_compression(self : Self) -> Int {
  self.spatial_compression
}

///|
pub fn MiniMaxH3VideoVaeConfig::temporal_compression(self : Self) -> Int {
  self.temporal_compression
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_layer_count(self : Self) -> Int {
  self.decoder_layer_count
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_attention_head_count(
  self : Self,
) -> Int {
  self.decoder_attention_head_count
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_attention_head_dimension(
  self : Self,
) -> Int {
  self.decoder_attention_head_dimension
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_hidden_size(self : Self) -> Int {
  self.decoder_attention_head_count * self.decoder_attention_head_dimension
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_register_token_count(
  self : Self,
) -> Int {
  self.decoder_register_token_count
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_ffn_multiplier(self : Self) -> Int {
  self.decoder_ffn_multiplier
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_ffn_dimension(self : Self) -> Int {
  self.decoder_hidden_size() * self.decoder_ffn_multiplier
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_rope_theta(self : Self) -> Double {
  self.decoder_rope_theta
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_rope_dimension_ratio(
  self : Self,
) -> Double {
  self.decoder_rope_dimension_ratio
}

///|
pub fn MiniMaxH3VideoVaeConfig::decoder_norm_epsilon(self : Self) -> Double {
  self.decoder_norm_epsilon
}

///|
/// Exact F32 coefficients used by the official decode-side inverse latent
/// normalization, ordered by latent channel.
pub fn MiniMaxH3VideoVaeConfig::latent_means(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.latent_means
}

///|
/// Exact F32 coefficients used by the official decode-side inverse latent
/// normalization, ordered by latent channel.
pub fn MiniMaxH3VideoVaeConfig::latent_standard_deviations(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.latent_standard_deviations
}

///|
pub fn MiniMaxH3AudioVaeConfig::latent_channels(self : Self) -> Int {
  self.latent_channels
}

///|
pub fn MiniMaxH3AudioVaeConfig::architecture_schema_version(self : Self) -> Int {
  self.architecture_schema_version
}

///|
pub fn MiniMaxH3AudioVaeConfig::encoder_dimension(self : Self) -> Int {
  self.encoder_dimension
}

///|
pub fn MiniMaxH3AudioVaeConfig::encoder_rates(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.encoder_rates
}

///|
pub fn MiniMaxH3AudioVaeConfig::latent_dimension(self : Self) -> Int {
  self.latent_dimension
}

///|
pub fn MiniMaxH3AudioVaeConfig::decoder_dimension(self : Self) -> Int {
  self.decoder_dimension
}

///|
pub fn MiniMaxH3AudioVaeConfig::decoder_rates(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.decoder_rates
}

///|
pub fn MiniMaxH3AudioVaeConfig::decoder_kernel_sizes(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.decoder_kernel_sizes
}

///|
pub fn MiniMaxH3AudioVaeConfig::attention_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
pub fn MiniMaxH3AudioVaeConfig::resblock_kernel_sizes(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.resblock_kernel_sizes
}

///|
pub fn MiniMaxH3AudioVaeConfig::resblock_dilation_sizes(
  self : Self,
) -> ReadOnlyArray[ReadOnlyArray[Int]] {
  self.resblock_dilation_sizes
}

///|
pub fn MiniMaxH3AudioVaeConfig::sampling_rate(self : Self) -> Int {
  self.sampling_rate
}

///|
pub fn MiniMaxH3AudioVaeConfig::output_channels(self : Self) -> Int {
  self.output_channels
}

///|
pub fn MiniMaxH3AudioVaeConfig::latent_rate(self : Self) -> Int {
  self.latent_rate
}

///|
pub fn MiniMaxH3AudioVaeConfig::latent_means(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.latent_means
}

///|
pub fn MiniMaxH3AudioVaeConfig::latent_standard_deviations(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.latent_standard_deviations
}

///|
pub fn MiniMaxH3SchedulerConfig::document(
  self : Self,
) -> MiniMaxH3ConfigDocument {
  self.document
}

///|
pub fn MiniMaxH3SchedulerConfig::shift(self : Self) -> Double {
  self.shift
}

///|
pub fn GlmDsaPooledKeys::pool_count(self : Self) -> Int32 {
  self.pool_count
}

///|
pub fn GlmDsaPooledKeys::pool_size(self : Self) -> Int32 {
  self.pool_size
}

///|
pub fn GlmDsaPooledKeys::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn GlmDsaPooledKeys::keys(self : Self) -> ReadOnlyArray[Double] {
  self.keys
}

///|
pub fn GlmDsaPooledKeys::raw_indices(self : Self) -> ReadOnlyArray[Int32] {
  self.raw_indices
}

///|
pub fn GlmDsaPooledKeys::valid(self : Self) -> ReadOnlyArray[Bool] {
  self.valid
}

///|
pub fn GlmDsaTopKState::source_layer(self : Self) -> Int32 {
  self.source_layer
}

///|
pub fn GlmDsaTopKState::last_consumer_layer(self : Self) -> Int32 {
  self.last_consumer_layer
}

///|
pub fn GlmDsaTopKState::key_count(self : Self) -> Int32 {
  self.key_count
}

///|
pub fn GlmDsaTopKState::indices(self : Self) -> ReadOnlyArray[Int32] {
  self.indices
}

///|
pub fn GlmKdaConvolutionResult::output(self : Self) -> ReadOnlyArray[Double] {
  self.output
}

///|
pub fn GlmKdaConvolutionResult::final_state(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.final_state
}

///|
pub fn GlmKdaConvolutionResult::sequence_length(self : Self) -> Int32 {
  self.sequence_length
}

///|
pub fn GlmKdaConvolutionResult::channel_count(self : Self) -> Int32 {
  self.channel_count
}

///|
pub fn GlmKdaRecurrentResult::output(self : Self) -> ReadOnlyArray[Double] {
  self.output
}

///|
pub fn GlmKdaRecurrentResult::final_state(self : Self) -> ReadOnlyArray[Double] {
  self.final_state
}

///|
pub fn GlmKdaRecurrentResult::sequence_length(self : Self) -> Int32 {
  self.sequence_length
}

///|
pub fn GlmKdaRecurrentResult::head_count(self : Self) -> Int32 {
  self.head_count
}

///|
pub fn GlmKdaRecurrentResult::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn GlmVisionReferenceResult::row_count(self : Self) -> Int32 {
  self.row_count
}

///|
pub fn GlmVisionReferenceResult::width(self : Self) -> Int32 {
  self.width
}

///|
pub fn GlmVisionReferenceResult::values(self : Self) -> ReadOnlyArray[Double] {
  self.values
}

///|
pub fn GlmReferenceLayerStep::layer_index(self : Self) -> Int32 {
  self.layer_index
}

///|
pub fn GlmReferenceLayerStep::attention(
  self : Self,
) -> GlmReferenceAttentionStep {
  self.attention
}

///|
pub fn GlmReferenceLayerStep::feed_forward(
  self : Self,
) -> GlmReferenceFeedForwardStep {
  self.feed_forward
}

///|
pub fn GlmReferenceLayerStep::has_hyper_connection(self : Self) -> Bool {
  self.hyper_connection
}

///|
pub fn GlmHybridReferenceSchedule::steps(
  self : Self,
) -> ReadOnlyArray[GlmReferenceLayerStep] {
  self.steps
}

///|
pub fn GlmHybridReferenceSchedule::dsa_compute_count(self : Self) -> Int32 {
  self.dsa_compute_count
}

///|
pub fn GlmHybridReferenceSchedule::dsa_reuse_count(self : Self) -> Int32 {
  self.dsa_reuse_count
}

///|
pub fn GlmHybridReferenceSchedule::kda_count(self : Self) -> Int32 {
  self.kda_count
}

///|
pub fn GlmHybridReferenceSchedule::dense_count(self : Self) -> Int32 {
  self.dense_count
}

///|
pub fn GlmHybridReferenceSchedule::moe_count(self : Self) -> Int32 {
  self.moe_count
}

///|
pub fn GlmHybridReferenceSchedule::hyper_connection_site_count(
  self : Self,
) -> Int32 {
  self.hyper_connection_site_count
}

///|
pub fn GlmHybridReferenceSchedule::vision_layer_count(self : Self) -> Int32 {
  self.vision_layer_count
}

///|
pub fn GlmSharedAdvancedFixture::moe_output(self : Self) -> Float {
  self.moe_output
}

///|
pub fn GlmSharedAdvancedFixture::mixed_hyper_streams(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.mixed_hyper_streams
}

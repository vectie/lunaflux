///|
priv struct AdvancedPlanCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  mut byte_count : Int
}

///|
fn AdvancedPlanCanonicalWriter::new() -> AdvancedPlanCanonicalWriter {
  { hasher: @canonical_sha256.CanonicalSha256::new(), byte_count: 0, }
}

///|
fn AdvancedPlanCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise AdvancedDecoderPlanError {
  if count < 0 || self.byte_count > 65536 - count {
    raise CanonicalPlanTooLarge
  }
  self.byte_count += count
}

///|
fn AdvancedPlanCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise AdvancedDecoderPlanError {
  self.reserve(1)
  if !self.hasher.update_byte(value) {
    abort("finalized advanced decoder plan hash")
  }
}

///|
fn AdvancedPlanCanonicalWriter::u64(
  self : Self,
  value : UInt64,
) -> Unit raise AdvancedDecoderPlanError {
  self.reserve(8)
  if !self.hasher.update_u64_le(value) {
    abort("finalized advanced decoder plan hash")
  }
}

///|
fn AdvancedPlanCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise AdvancedDecoderPlanError {
  self.u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn AdvancedPlanCanonicalWriter::double(
  self : Self,
  value : Double,
) -> Unit raise AdvancedDecoderPlanError {
  self.u64(value.reinterpret_as_uint64())
}

///|
fn AdvancedPlanCanonicalWriter::text(
  self : Self,
  value : String,
) -> Unit raise AdvancedDecoderPlanError {
  let bytes = @utf8.encode(value)
  self.int(bytes.length())
  self.reserve(bytes.length())
  if !self.hasher.update_bytes(bytes) {
    abort("finalized advanced decoder plan hash")
  }
}

///|
fn encode_shape(
  writer : AdvancedPlanCanonicalWriter,
  shape : AdvancedDecoderShape,
) -> Unit raise AdvancedDecoderPlanError {
  writer.int(shape.vocabulary_size())
  writer.int(shape.hidden_size())
  writer.int(shape.decoder_layer_count())
  writer.int(shape.attention_head_count())
  writer.int(shape.kv_head_count())
  writer.int(shape.attention_head_dimension())
  writer.int(shape.rope_head_dimension())
  writer.int(shape.query_low_rank())
  writer.int(shape.output_low_rank())
  writer.int(shape.output_group_count())
  writer.int(shape.maximum_position_count())
}

///|
fn encode_attention(
  writer : AdvancedPlanCanonicalWriter,
  attention : AdvancedAttentionPlan,
) -> Unit raise AdvancedDecoderPlanError {
  match attention {
    SlidingWindowCausal(window_size~) => {
      writer.byte(0x01)
      writer.int(window_size)
    }
    IndexedCompressed(
      compression_ratio~,
      index_head_count~,
      index_head_dimension~,
      index_top_k~
    ) => {
      writer.byte(0x02)
      writer.int(compression_ratio)
      writer.int(index_head_count)
      writer.int(index_head_dimension)
      writer.int(index_top_k)
    }
    DeterministicCompressed(compression_ratio~, window_size~) => {
      writer.byte(0x03)
      writer.int(compression_ratio)
      writer.int(window_size)
    }
  }
}

///|
fn encode_auxiliary(
  writer : AdvancedPlanCanonicalWriter,
  auxiliary : AuxiliaryPredictionPlan,
) -> Unit raise AdvancedDecoderPlanError {
  match auxiliary {
    MultiTokenPrediction(layer_count~) => {
      writer.byte(0x01)
      writer.int(layer_count)
    }
    MultiTokenPredictionWithDspark(layer_count~, dspark~) => {
      writer.byte(0x02)
      writer.int(layer_count)
      writer.int(dspark.block_size())
      writer.int(dspark.noise_token_id())
      writer.int(dspark.target_layer_ids().length())
      for layer in dspark.target_layer_ids() {
        writer.int(layer)
      }
      writer.int(dspark.markov_rank())
    }
  }
}

///|
fn canonical_plan_digest(
  schema_version : Int,
  profile_identity : String,
  content_digest : @spec.ContentDigest,
  shape : AdvancedDecoderShape,
  low_rank_attention : LowRankAttentionPlan,
  position_encoding : PositionEncodingPlan,
  hyper_connection : HyperConnectionPlan,
  moe : MoeExecutionPlan,
  layers : ArrayView[AdvancedDecoderLayerPlan],
  auxiliary_prediction : AuxiliaryPredictionPlan,
) -> @spec.PlanDigest raise AdvancedDecoderPlanError {
  let writer = AdvancedPlanCanonicalWriter::new()
  writer.text("lunaflux/advanced-decoder-execution-plan/v1")
  writer.int(schema_version)
  writer.text(profile_identity)
  writer.text(content_digest.as_hex())
  encode_shape(writer, shape)
  writer.int(low_rank_attention.query_rank())
  writer.int(low_rank_attention.output_rank())
  writer.int(low_rank_attention.output_group_count())
  writer.double(low_rank_attention.query_key_norm_epsilon())
  writer.double(position_encoding.rope_theta())
  writer.double(position_encoding.compressed_rope_theta())
  writer.int(position_encoding.original_position_count())
  writer.double(position_encoding.scaling_factor())
  writer.double(position_encoding.beta_fast())
  writer.double(position_encoding.beta_slow())
  writer.int(hyper_connection.width())
  writer.double(hyper_connection.control_epsilon())
  writer.double(hyper_connection.normalization_epsilon())
  writer.int(hyper_connection.sinkhorn_iterations())
  writer.int(moe.routed_expert_count())
  writer.int(moe.shared_expert_count())
  writer.int(moe.experts_per_token())
  writer.int(moe.intermediate_size())
  writer.double(moe.routed_scaling_factor())
  writer.byte(if moe.normalize_selected_probabilities() { 1 } else { 0 })
  writer.byte(
    match moe.score_function() {
      SqrtSoftplus => 0x01
      Sigmoid => 0x02
    },
  )
  match moe.selection_method() {
    NoAuxiliaryLossTokenChoice => writer.byte(0x01)
    TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~) => {
      writer.byte(0x02)
      writer.int(hash_layer_count)
    }
  }
  match moe.activation() {
    SwiGlu => writer.byte(0x01)
    ClampedSwiGlu(limit) => {
      writer.byte(0x02)
      writer.double(limit)
    }
  }
  writer.int(layers.length())
  for layer in layers {
    writer.int(layer.layer_index())
    encode_attention(writer, layer.attention())
  }
  encode_auxiliary(writer, auxiliary_prediction)
  try! @spec.PlanDigest::from_sha256(writer.hasher.finalize_hex())
}

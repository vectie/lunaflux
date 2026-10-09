///|
priv struct GlmPlanCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
}

///|
fn GlmPlanCanonicalWriter::new() -> GlmPlanCanonicalWriter {
  { hasher: @canonical_sha256.CanonicalSha256::new(), }
}

///|
fn GlmPlanCanonicalWriter::byte(self : Self, value : Byte) -> Unit {
  if !self.hasher.update_byte(value) {
    abort("finalized GLM hybrid plan canonical hash")
  }
}

///|
fn GlmPlanCanonicalWriter::bytes(self : Self, value : BytesView) -> Unit {
  if !self.hasher.update_bytes(value) {
    abort("finalized GLM hybrid plan canonical hash")
  }
}

///|
fn GlmPlanCanonicalWriter::int(self : Self, value : Int) -> Unit {
  if !self.hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()) {
    abort("finalized GLM hybrid plan canonical hash")
  }
}

///|
fn GlmPlanCanonicalWriter::double(self : Self, value : Double) -> Unit {
  if !self.hasher.update_u64_le(value.reinterpret_as_uint64()) {
    abort("finalized GLM hybrid plan canonical hash")
  }
}

///|
fn GlmPlanCanonicalWriter::bool(self : Self, value : Bool) -> Unit {
  self.byte(if value { 0x01 } else { 0x00 })
}

///|
fn GlmPlanCanonicalWriter::string(self : Self, value : StringView) -> Unit {
  let encoded = @utf8.encode(value)
  self.int(encoded.length())
  self.bytes(encoded)
}

///|
fn encode_sparse_attention(
  writer : GlmPlanCanonicalWriter,
  attention : GlmDynamicSparseAttentionPlan,
) -> Unit {
  writer.int(attention.query_head_count())
  writer.int(attention.kv_head_count())
  writer.int(attention.query_lora_rank())
  writer.int(attention.kv_lora_rank())
  writer.int(attention.qk_nope_head_dimension())
  writer.int(attention.qk_rope_head_dimension())
  writer.int(attention.value_head_dimension())
  writer.double(attention.rms_norm_epsilon())
  match attention.rope_theta() {
    Some(theta) => {
      writer.byte(0x01)
      writer.double(theta)
    }
    None => writer.byte(0x00)
  }
  writer.bool(attention.rope_interleave())
  writer.int(attention.index_head_count())
  writer.int(attention.index_head_dimension())
  writer.int(attention.index_topk())
  writer.bool(attention.indexer_rope_interleave())
  writer.bool(attention.index_share_for_mtp_iteration())
  match attention.index_pooling() {
    Some(pooling) => {
      writer.byte(0x01)
      writer.int(pooling.kernel_size())
      writer.bool(pooling.compress())
      writer.bool(pooling.always_select_tail())
    }
    None => writer.byte(0x00)
  }
}

///|
fn encode_kda(
  writer : GlmPlanCanonicalWriter,
  kda : GlmKdaExecutionPlan?,
) -> Unit {
  match kda {
    Some(plan) => {
      writer.byte(0x01)
      writer.int(plan.head_count())
      writer.int(plan.head_dimension())
      writer.int(plan.short_convolution_kernel_size())
      writer.double(plan.gate_lower_bound())
    }
    None => writer.byte(0x00)
  }
}

///|
fn encode_hyper_connection(
  writer : GlmPlanCanonicalWriter,
  hyper : @advanced_decoder_execution_plan.HyperConnectionPlan?,
) -> Unit {
  match hyper {
    Some(plan) => {
      writer.byte(0x01)
      writer.int(plan.width())
      writer.double(plan.epsilon())
      writer.int(plan.sinkhorn_iterations())
    }
    None => writer.byte(0x00)
  }
}

///|
fn encode_moe(
  writer : GlmPlanCanonicalWriter,
  moe : @advanced_decoder_execution_plan.MoeExecutionPlan,
) -> Unit {
  writer.int(moe.routed_expert_count())
  writer.int(moe.shared_expert_count())
  writer.int(moe.experts_per_token())
  writer.int(moe.intermediate_size())
  writer.double(moe.routed_scaling_factor())
  writer.bool(moe.normalize_selected_probabilities())
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
}

///|
fn encode_layers(
  writer : GlmPlanCanonicalWriter,
  layers : ArrayView[GlmHybridLayerPlan],
) -> Unit {
  writer.int(layers.length())
  for layer in layers {
    writer.int(layer.layer_index())
    writer.byte(
      match layer.attention() {
        DynamicSparse(ComputeTopK) => 0x01
        DynamicSparse(ReusePreviousTopK) => 0x02
        KimiDelta => 0x03
      },
    )
    writer.byte(
      match layer.feed_forward() {
        DenseSwiGlu => 0x01
        RoutedMoe => 0x02
      },
    )
  }
}

///|
fn encode_auxiliary(
  writer : GlmPlanCanonicalWriter,
  auxiliary : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan,
) -> Unit {
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
      writer.int(dspark.markov_rank())
      writer.int(dspark.target_layer_ids().length())
      for layer in dspark.target_layer_ids() {
        writer.int(layer)
      }
    }
  }
}

///|
fn encode_vision(
  writer : GlmPlanCanonicalWriter,
  vision : GlmVisionConditioningPlan?,
) -> Unit {
  match vision {
    Some(plan) => {
      writer.byte(0x01)
      writer.int(plan.layer_count())
      writer.int(plan.hidden_size())
      writer.int(plan.intermediate_size())
      writer.int(plan.attention_head_count())
      writer.int(plan.spatial_patch_size())
      writer.int(plan.temporal_patch_size())
      writer.int(plan.spatial_merge_size())
      writer.int(plan.decoder_projection_size())
      writer.int(plan.projection_intermediate_size())
      writer.int(plan.image_token_id())
      writer.int(plan.video_token_id())
      writer.int(plan.image_start_token_id())
      writer.int(plan.image_end_token_id())
      writer.int(plan.video_start_token_id())
      writer.int(plan.video_end_token_id())
    }
    None => writer.byte(0x00)
  }
}

///|
fn canonical_plan_digest(
  content_digest : @spec.ContentDigest,
  schema_version : Int,
  profile_identity : StringView,
  architecture : StringView,
  model_type : StringView,
  vocabulary_size : Int,
  hidden_size : Int,
  dense_intermediate_size : Int,
  decoder_layer_count : Int,
  context_length : Int,
  sparse_attention : GlmDynamicSparseAttentionPlan,
  kda : GlmKdaExecutionPlan?,
  hyper : @advanced_decoder_execution_plan.HyperConnectionPlan?,
  moe : @advanced_decoder_execution_plan.MoeExecutionPlan,
  moe_router_float32 : Bool,
  moe_correction_bias : Bool,
  layers : ArrayView[GlmHybridLayerPlan],
  auxiliary : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan,
  vision : GlmVisionConditioningPlan?,
) -> @spec.PlanDigest {
  let writer = GlmPlanCanonicalWriter::new()
  writer.bytes(b"lunaflux.glm-hybrid-execution-plan.v1\x00")
  writer.string(content_digest.as_hex())
  writer.int(schema_version)
  writer.string(profile_identity)
  writer.string(architecture)
  writer.string(model_type)
  writer.int(vocabulary_size)
  writer.int(hidden_size)
  writer.int(dense_intermediate_size)
  writer.int(decoder_layer_count)
  writer.int(context_length)
  encode_sparse_attention(writer, sparse_attention)
  encode_kda(writer, kda)
  encode_hyper_connection(writer, hyper)
  encode_moe(writer, moe)
  writer.bool(moe_router_float32)
  writer.bool(moe_correction_bias)
  encode_layers(writer, layers)
  encode_auxiliary(writer, auxiliary)
  encode_vision(writer, vision)
  try! @spec.PlanDigest::from_sha256(writer.hasher.finalize_hex())
}

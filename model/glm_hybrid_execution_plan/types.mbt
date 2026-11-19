///|
/// Per-layer DSA index authority. A shared layer consumes the most recently
/// computed index selection and may not appear before a computing layer.
pub(all) enum GlmSparseIndexAction {
  ComputeTopK
  ReusePreviousTopK
} derive(Debug, Eq)

///|
/// Optional learned compression before DSA top-k selection.
pub struct GlmSparseIndexPoolingPlan {
  priv kernel_size : Int32
  priv compress : Bool
  priv always_select_tail : Bool
} derive(Debug, Eq)

///|
/// Complete GLM low-rank dynamic sparse-attention geometry shared by DSA
/// layers in one profile.
pub struct GlmDynamicSparseAttentionPlan {
  priv query_head_count : Int32
  priv kv_head_count : Int32
  priv query_lora_rank : Int32
  priv kv_lora_rank : Int32
  priv qk_nope_head_dimension : Int32
  priv qk_rope_head_dimension : Int32
  priv value_head_dimension : Int32
  priv rms_norm_epsilon : Double
  priv rope_theta : Double?
  priv rope_interleave : Bool
  priv index_head_count : Int32
  priv index_head_dimension : Int32
  priv index_topk : Int32
  priv indexer_rope_interleave : Bool
  priv index_share_for_mtp_iteration : Bool
  priv index_pooling : GlmSparseIndexPoolingPlan?
} derive(Debug, Eq)

///|
/// Kimi Delta Attention recurrent-state geometry. The convolution history and
/// delta state are distinct from a decoder KV cache.
pub struct GlmKdaExecutionPlan {
  priv head_count : Int32
  priv head_dimension : Int32
  priv short_convolution_kernel_size : Int32
  priv gate_lower_bound : Double
} derive(Debug, Eq)

///|
/// Closed per-layer attention schedule for GLM hybrid decoders.
pub(all) enum GlmLayerAttentionPlan {
  DynamicSparse(GlmSparseIndexAction)
  KimiDelta
} derive(Debug, Eq)

///|
/// Closed per-layer feed-forward schedule. Exact dense and routed geometry is
/// retained once at the complete-plan level.
pub(all) enum GlmLayerFeedForwardPlan {
  DenseSwiGlu
  RoutedMoe
} derive(Debug, Eq)

///|
/// One immutable decoder layer in execution order.
pub struct GlmHybridLayerPlan {
  priv layer_index : Int32
  priv attention : GlmLayerAttentionPlan
  priv feed_forward : GlmLayerFeedForwardPlan
} derive(Debug, Eq)

///|
/// Exact vision frontend and special-token contract for a conditional GLM
/// decoder. This grants no image decoding or preprocessing authority.
pub struct GlmVisionConditioningPlan {
  priv layer_count : Int32
  priv hidden_size : Int32
  priv intermediate_size : Int32
  priv attention_head_count : Int32
  priv spatial_patch_size : Int32
  priv temporal_patch_size : Int32
  priv spatial_merge_size : Int32
  priv decoder_projection_size : Int32
  priv projection_intermediate_size : Int32
  priv image_token_id : Int32
  priv video_token_id : Int32
  priv image_start_token_id : Int32
  priv image_end_token_id : Int32
  priv video_start_token_id : Int32
  priv video_end_token_id : Int32
} derive(Debug, Eq)

///|
/// Public, payload-safe validation failures.
pub(all) suberror GlmHybridExecutionPlanError {
  InvalidSchemaVersion
  InvalidProfileIdentity
  InvalidModelGeometry
  InvalidSparseAttentionGeometry
  InvalidKdaGeometry
  InvalidLayerOrder
  LayerCountMismatch
  InvalidSparseIndexSchedule
  MissingKdaContract
  UnusedKdaContract
  MissingDenseLayer
  MissingRoutedMoeLayer
  InvalidVisionConditioning
  InvalidAuxiliaryPrediction
} derive(Debug, Eq)

///|
/// Validated GLM hybrid decoder plan. Shared MoE, mHC, and MTP vocabulary is
/// imported from the architecture-neutral advanced decoder package; only
/// DSA-sharing, KDA, layer mixing, and vision conditioning are GLM-owned here.
pub struct GlmHybridExecutionPlan {
  priv identity : @spec.ModelIdentity
  priv schema_version : Int32
  priv profile_identity : String
  priv architecture : String
  priv model_type : String
  priv vocabulary_size : Int32
  priv hidden_size : Int32
  priv dense_intermediate_size : Int32
  priv decoder_layer_count : Int32
  priv context_length : Int32
  priv sparse_attention : GlmDynamicSparseAttentionPlan
  priv kda : GlmKdaExecutionPlan?
  priv hyper_connection : @advanced_decoder_execution_plan.HyperConnectionPlan?
  priv moe : @advanced_decoder_execution_plan.MoeExecutionPlan
  priv moe_router_float32 : Bool
  priv moe_correction_bias : Bool
  priv layers : ReadOnlyArray[GlmHybridLayerPlan]
  priv auxiliary_prediction : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan
  priv vision_conditioning : GlmVisionConditioningPlan?
} derive(Debug, Eq)

///|
pub fn full_schema(
  metadata : @glm53_spec.Glm53FullMetadata,
  encoding : Glm53WeightEncoding,
) -> Glm53WeightSchema {
  { profile: FullText, content_digest: metadata.content_digest(), encoding, }
}

///|
pub fn flash_schema(
  metadata : @glm53_spec.Glm53FlashMetadata,
  encoding : Glm53WeightEncoding,
) -> Glm53WeightSchema {
  {
    profile: FlashMultimodal,
    content_digest: metadata.content_digest(),
    encoding,
  }
}

///|
fn Glm53WeightSchema::hidden_size(self : Self) -> Int {
  match self.profile {
    FullText => @glm53_spec.Glm53FullSpec::official().hidden_size()
    FlashMultimodal => @glm53_spec.Glm53FlashSpec::official().hidden_size()
  }
}

///|
fn Glm53WeightSchema::vocabulary_size(self : Self) -> Int {
  match self.profile {
    FullText => @glm53_spec.Glm53FullSpec::official().vocabulary_size()
    FlashMultimodal => @glm53_spec.Glm53FlashSpec::official().vocabulary_size()
  }
}

///|
fn Glm53WeightSchema::layer_count(self : Self) -> Int {
  match self.profile {
    FullText => @glm53_spec.Glm53FullSpec::official().layer_count()
    FlashMultimodal => @glm53_spec.Glm53FlashSpec::official().layer_count()
  }
}

///|
pub fn Glm53WeightSchema::global_shape(
  self : Self,
  role : Glm53GlobalTensorRole,
) -> ReadOnlyArray[Int64] {
  let hidden = Int64::from_int(self.hidden_size())
  let vocabulary = Int64::from_int(self.vocabulary_size())
  ReadOnlyArray::from_array(
    match role {
      TokenEmbedding | LanguageModelHead => [vocabulary, hidden]
      FinalNorm => [hidden]
    },
  )
}

///|
fn Glm53WeightSchema::layer_semantics(
  self : Self,
  layer : Int,
) -> (@glm53_spec.Glm53AttentionKind, @glm53_spec.Glm53FeedForwardKind) raise Glm53WeightsError {
  if layer < 0 || layer >= self.layer_count() {
    raise InvalidLayerIndex
  }
  match self.profile {
    FullText => {
      let spec = @glm53_spec.Glm53FullSpec::official()
      (try! spec.attention_kind(layer), try! spec.feed_forward_kind(layer))
    }
    FlashMultimodal => {
      let spec = @glm53_spec.Glm53FlashSpec::official()
      (try! spec.attention_kind(layer), try! spec.feed_forward_kind(layer))
    }
  }
}

///|
fn is_dense_role(role : Glm53TextTensorRole) -> Bool {
  role is (DenseGate | DenseUp | DenseDown)
}

///|
fn is_moe_role(role : Glm53TextTensorRole) -> Bool {
  role
  is (MoeRouter
  | MoeCorrectionBias
  | RoutedGateUp
  | RoutedDown
  | SharedGate
  | SharedUp
  | SharedDown)
}

///|
fn is_dsa_role(role : Glm53TextTensorRole) -> Bool {
  role
  is (QueryA
  | QueryANorm
  | QueryB
  | KeyValueA
  | KeyValueANorm
  | KeyValueB
  | AttentionOutput
  | IndexQuery
  | IndexKey
  | IndexKeyNorm
  | IndexKeyNormBias
  | IndexWeights
  | IndexPoolPosition
  | IndexPoolGate)
}

///|
fn is_linear_role(role : Glm53TextTensorRole) -> Bool {
  role
  is (LinearQuery
  | LinearKey
  | LinearValue
  | LinearConvolution
  | LinearBeta
  | LinearGateA
  | LinearGateB
  | LinearOutputNorm
  | LinearOutput
  | LinearForgetA
  | LinearForgetB
  | LinearForgetBias
  | LinearForgetLogA)
}

///|
fn is_hyper_role(role : Glm53TextTensorRole) -> Bool {
  role
  is (HyperAttentionFunction
  | HyperAttentionBase
  | HyperAttentionScale
  | HyperFfnFunction
  | HyperFfnBase
  | HyperFfnScale)
}

///|
fn Glm53WeightSchema::validate_role(
  self : Self,
  layer : Int,
  role : Glm53TextTensorRole,
) -> Unit raise Glm53WeightsError {
  let (attention, feed_forward) = self.layer_semantics(layer)
  if is_dense_role(role) && feed_forward != DenseSwiGlu {
    raise TensorRoleNotPresent(role)
  }
  if is_moe_role(role) && feed_forward != RoutedMixtureOfExperts {
    raise TensorRoleNotPresent(role)
  }
  if is_dsa_role(role) && attention != DeepSeekSparseAttention {
    raise TensorRoleNotPresent(role)
  }
  if is_dsa_role(role) && self.profile == FullText {
    let full = @glm53_spec.Glm53FullSpec::official()
    let computes_index_topk = try! full.computes_index_topk(layer)
    if role
      is (IndexQuery
      | IndexKey
      | IndexKeyNorm
      | IndexKeyNormBias
      | IndexWeights
      | IndexPoolPosition
      | IndexPoolGate) &&
      !computes_index_topk {
      raise TensorRoleNotPresent(role)
    }
  }
  if role is (IndexPoolPosition | IndexPoolGate) &&
    self.profile != FlashMultimodal {
    raise TensorRoleNotPresent(role)
  }
  if is_linear_role(role) && attention != KimiDeltaAttention {
    raise TensorRoleNotPresent(role)
  }
  if is_hyper_role(role) && self.profile != FlashMultimodal {
    raise TensorRoleNotPresent(role)
  }
}

///|
/// Returns the exact logical tensor shape for one official text-layer role.
/// Physical FP8 scale tensors are deliberately outside this semantic shape;
/// the encoding accessor fixes their 128x128 block policy.
pub fn Glm53WeightSchema::text_layer_shape(
  self : Self,
  layer : Int,
  role : Glm53TextTensorRole,
) -> ReadOnlyArray[Int64] raise Glm53WeightsError {
  self.validate_role(layer, role)
  let hidden = Int64::from_int(self.hidden_size())
  let (
    dense,
    moe,
    experts,
    heads,
    q_rank,
    kv_rank,
    nope,
    rope,
    value,
    index_heads,
    index_dim,
  ) = match self.profile {
    FullText => {
      let spec = @glm53_spec.Glm53FullSpec::official()
      (
        spec.dense_intermediate_size(),
        spec.moe_intermediate_size(),
        spec.routed_expert_count(),
        spec.attention_head_count(),
        spec.q_lora_rank(),
        spec.kv_lora_rank(),
        spec.qk_nope_head_dimension(),
        spec.qk_rope_head_dimension(),
        spec.value_head_dimension(),
        spec.index_head_count(),
        spec.index_head_dimension(),
      )
    }
    FlashMultimodal => {
      let spec = @glm53_spec.Glm53FlashSpec::official()
      (
        spec.dense_intermediate_size(),
        spec.moe_intermediate_size(),
        spec.routed_expert_count(),
        spec.attention_head_count(),
        spec.q_lora_rank(),
        spec.kv_lora_rank(),
        spec.qk_nope_head_dimension(),
        spec.qk_rope_head_dimension(),
        spec.value_head_dimension(),
        spec.index_head_count(),
        spec.index_head_dimension(),
      )
    }
  }
  let dense64 = Int64::from_int(dense)
  let moe64 = Int64::from_int(moe)
  let experts64 = Int64::from_int(experts)
  let heads64 = Int64::from_int(heads)
  let q_rank64 = Int64::from_int(q_rank)
  let kv_rank64 = Int64::from_int(kv_rank)
  let qk64 = Int64::from_int(nope + rope)
  let rope64 = Int64::from_int(rope)
  let value64 = Int64::from_int(value)
  let index_heads64 = Int64::from_int(index_heads)
  let index64 = Int64::from_int(index_dim)
  let linear_heads = 64L
  let linear_dim = 128L
  let linear_width = linear_heads * linear_dim
  let hyper_streams = 4L
  let hyper_mix = (2L + hyper_streams) * hyper_streams
  let shape : Array[Int64] = match role {
    InputNorm | PostAttentionNorm => [hidden]
    DenseGate | DenseUp => [dense64, hidden]
    DenseDown => [hidden, dense64]
    MoeRouter => [experts64, hidden]
    MoeCorrectionBias => [experts64]
    RoutedGateUp => [experts64, 2L * moe64, hidden]
    RoutedDown => [experts64, hidden, moe64]
    SharedGate | SharedUp => [moe64, hidden]
    SharedDown => [hidden, moe64]
    QueryA => [q_rank64, hidden]
    QueryANorm => [q_rank64]
    QueryB => [heads64 * qk64, q_rank64]
    KeyValueA => [kv_rank64 + rope64, hidden]
    KeyValueANorm => [kv_rank64]
    KeyValueB => [heads64 * (Int64::from_int(nope) + value64), kv_rank64]
    AttentionOutput => [hidden, heads64 * value64]
    IndexQuery => [index_heads64 * index64, q_rank64]
    IndexKey => [index64, hidden]
    IndexKeyNorm | IndexKeyNormBias => [index64]
    IndexWeights => [index_heads64, hidden]
    IndexPoolPosition => [4L, index64]
    IndexPoolGate => [index64, hidden]
    LinearQuery | LinearKey | LinearValue => [linear_width, hidden]
    LinearConvolution => [3L * linear_width, 1L, 4L]
    LinearBeta => [linear_heads, hidden]
    LinearGateA | LinearForgetA => [linear_dim, hidden]
    LinearGateB | LinearForgetB => [linear_width, linear_dim]
    LinearOutputNorm => [linear_dim]
    LinearOutput => [hidden, linear_width]
    LinearForgetBias => [linear_width]
    LinearForgetLogA => [linear_heads]
    HyperAttentionFunction | HyperFfnFunction =>
      [hyper_mix, hyper_streams * hidden]
    HyperAttentionBase | HyperFfnBase => [hyper_mix]
    HyperAttentionScale | HyperFfnScale => [3L]
  }
  ReadOnlyArray::from_array(shape)
}

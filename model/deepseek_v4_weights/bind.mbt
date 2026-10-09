///|
fn parse_index(text : StringView) -> Int? {
  if text.is_empty() {
    return None
  }
  let mut value = 0
  for char in text {
    if char < '0' || char > '9' || value > 214748364 {
      return None
    }
    value = value * 10 + char.to_int() - '0'.to_int()
  }
  Some(value)
}

///|
fn suffix(parts : Array[StringView], start : Int) -> String {
  let builder = StringBuilder()
  for index in start..<parts.length() {
    if index > start {
      builder.write_char('.')
    }
    builder.write_view(parts[index])
  }
  builder.to_string()
}

///|
fn quantized_part(text : StringView) -> DeepSeekV4QuantizedPart? {
  match text {
    "weight" => Some(QuantizedWeight)
    "scale" => Some(QuantizationScale)
    _ => None
  }
}

///|
fn expert_matrix(text : StringView) -> DeepSeekV4ExpertMatrix? {
  match text {
    "w1" => Some(ExpertW1)
    "w2" => Some(ExpertW2)
    "w3" => Some(ExpertW3)
    _ => None
  }
}

///|
fn fixed_role(name : StringView) -> DeepSeekV4TensorRole? {
  match name {
    "attn.attn_sink" => Some(Attention(AttentionSink))
    "attn.q_norm.weight" => Some(Attention(QueryNorm))
    "attn.kv_norm.weight" => Some(Attention(KeyValueNorm))
    "attn.wq_a.weight" => Some(Attention(QueryA(QuantizedWeight)))
    "attn.wq_a.scale" => Some(Attention(QueryA(QuantizationScale)))
    "attn.wq_b.weight" => Some(Attention(QueryB(QuantizedWeight)))
    "attn.wq_b.scale" => Some(Attention(QueryB(QuantizationScale)))
    "attn.wkv.weight" => Some(Attention(KeyValue(QuantizedWeight)))
    "attn.wkv.scale" => Some(Attention(KeyValue(QuantizationScale)))
    "attn.wo_a.weight" => Some(Attention(OutputA(QuantizedWeight)))
    "attn.wo_a.scale" => Some(Attention(OutputA(QuantizationScale)))
    "attn.wo_b.weight" => Some(Attention(OutputB(QuantizedWeight)))
    "attn.wo_b.scale" => Some(Attention(OutputB(QuantizationScale)))
    "attn_norm.weight" => Some(AttentionNorm)
    "ffn.gate.weight" => Some(RouterWeight)
    "ffn.gate.bias" => Some(RouterBias)
    "ffn.gate.tid2eid" => Some(TokenExpertHash)
    "ffn_norm.weight" => Some(FfnNorm)
    "hc_attn_base" => Some(AttentionHyperConnection(HyperBase))
    "hc_attn_fn" => Some(AttentionHyperConnection(HyperFunction))
    "hc_attn_scale" => Some(AttentionHyperConnection(HyperScale))
    "hc_ffn_base" => Some(FfnHyperConnection(HyperBase))
    "hc_ffn_fn" => Some(FfnHyperConnection(HyperFunction))
    "hc_ffn_scale" => Some(FfnHyperConnection(HyperScale))
    "attn.compressor.ape" => Some(AttentionCompressor(CompressorApe))
    "attn.compressor.norm.weight" => Some(AttentionCompressor(CompressorNorm))
    "attn.compressor.wgate.weight" => Some(AttentionCompressor(CompressorGate))
    "attn.compressor.wkv.weight" =>
      Some(AttentionCompressor(CompressorKeyValue))
    "attn.indexer.compressor.ape" => Some(AttentionIndexer(IndexCompressorApe))
    "attn.indexer.compressor.norm.weight" =>
      Some(AttentionIndexer(IndexCompressorNorm))
    "attn.indexer.compressor.wgate.weight" =>
      Some(AttentionIndexer(IndexCompressorGate))
    "attn.indexer.compressor.wkv.weight" =>
      Some(AttentionIndexer(IndexCompressorKeyValue))
    "attn.indexer.weights_proj.weight" =>
      Some(AttentionIndexer(IndexWeightsProjection))
    "attn.indexer.wq_b.weight" =>
      Some(AttentionIndexer(IndexQuery(QuantizedWeight)))
    "attn.indexer.wq_b.scale" =>
      Some(AttentionIndexer(IndexQuery(QuantizationScale)))
    "emb.tok_emb.weight" => Some(MtpTokenEmbedding)
    "head.weight" => Some(MtpOutputHead)
    "enorm.weight" => Some(MtpEmbeddingNorm)
    "hnorm.weight" => Some(MtpHiddenNorm)
    "e_proj.weight" => Some(MtpEmbeddingProjection(QuantizedWeight))
    "e_proj.scale" => Some(MtpEmbeddingProjection(QuantizationScale))
    "h_proj.weight" => Some(MtpHiddenProjection(QuantizedWeight))
    "h_proj.scale" => Some(MtpHiddenProjection(QuantizationScale))
    "norm.weight" => Some(MtpFinalNorm)
    "hc_head_base" => Some(HeadHyperConnection(HyperBase))
    "hc_head_fn" => Some(HeadHyperConnection(HyperFunction))
    "hc_head_scale" => Some(HeadHyperConnection(HyperScale))
    "main_norm.weight" => Some(DsparkMainNorm)
    "main_proj.weight" => Some(DsparkMainProjection(QuantizedWeight))
    "main_proj.scale" => Some(DsparkMainProjection(QuantizationScale))
    "confidence_head.proj.weight" => Some(DsparkConfidenceProjection)
    "markov_head.markov_w1.weight" => Some(DsparkMarkovW1)
    "markov_head.markov_w2.weight" => Some(DsparkMarkovW2)
    _ => None
  }
}

///|
fn parse_semantic(
  name : StringView,
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4TensorSemantic? {
  match name {
    "embed.weight" => return Some({ scope: Global, role: TokenEmbedding, })
    "head.weight" => return Some({ scope: Global, role: OutputHead, })
    "norm.weight" => return Some({ scope: Global, role: FinalNorm, })
    "hc_head_base" =>
      return Some({ scope: Global, role: HeadHyperConnection(HyperBase), })
    "hc_head_fn" =>
      return Some({ scope: Global, role: HeadHyperConnection(HyperFunction), })
    "hc_head_scale" =>
      return Some({ scope: Global, role: HeadHyperConnection(HyperScale), })
    _ => ()
  }
  let parts = name.split(".").to_array()
  if parts.length() < 3 {
    return None
  }
  let (scope, start) = match parts[0] {
    "layers" =>
      match parse_index(parts[1]) {
        Some(index) if index < spec.layer_count() => (DecoderLayer(index), 2)
        _ => return None
      }
    "mtp" =>
      match parse_index(parts[1]) {
        Some(index) if index < spec.prediction_stage_count() =>
          (PredictionStage(index), 2)
        _ => return None
      }
    _ => return None
  }
  if parts.length() == start + 5 &&
    parts[start] == "ffn" &&
    parts[start + 1] == "experts" {
    match
      (
        parse_index(parts[start + 2]),
        expert_matrix(parts[start + 3]),
        quantized_part(parts[start + 4]),
      ) {
      (Some(expert), Some(matrix), Some(part)) if expert <
        spec.routed_expert_count() =>
        return Some({ scope, role: RoutedExpert(expert, matrix, part), })
      _ => return None
    }
  }
  if parts.length() == start + 4 &&
    parts[start] == "ffn" &&
    parts[start + 1] == "shared_experts" {
    match (expert_matrix(parts[start + 2]), quantized_part(parts[start + 3])) {
      (Some(matrix), Some(part)) =>
        return Some({ scope, role: SharedExpert(matrix, part), })
      _ => return None
    }
  }
  fixed_role(suffix(parts, start)).map(role => { scope, role, })
}

///|
fn shape2(a : Int, b : Int) -> ReadOnlyArray[Int64] {
  [a.to_int64(), b.to_int64()]
}

///|
fn shape1(a : Int) -> ReadOnlyArray[Int64] {
  [a.to_int64()]
}

///|
fn scale2(a : Int, b : Int) -> ReadOnlyArray[Int64] {
  shape2((a + 127) / 128, (b + 127) / 128)
}

///|
fn expert_shape(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  matrix : DeepSeekV4ExpertMatrix,
  part : DeepSeekV4QuantizedPart,
  routed : Bool,
) -> (
  DeepSeekV4SafetensorsDType,
  ReadOnlyArray[Int64],
  DeepSeekV4WeightEncoding,
) {
  let h = spec.hidden_size()
  let m = spec.moe_intermediate_size()
  let (rows, columns) = match matrix {
    ExpertW1 | ExpertW3 => (m, h)
    ExpertW2 => (h, m)
  }
  if part is QuantizationScale {
    if routed && spec.expert_storage() is Fp4E2M1 {
      (F8E8M0, shape2(rows, columns / 32), Ue8M0Scale)
    } else {
      (F8E8M0, scale2(rows, columns), Ue8M0Scale)
    }
  } else if routed && spec.expert_storage() is Fp4E2M1 {
    (I8, shape2(rows, columns / 2), PackedFp4E2M1Block32)
  } else {
    (F8E4M3, shape2(rows, columns), Fp8E4M3Block128)
  }
}

///|
fn expected_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  semantic : DeepSeekV4TensorSemantic,
) -> (
  DeepSeekV4SafetensorsDType,
  ReadOnlyArray[Int64],
  DeepSeekV4WeightEncoding,
)? {
  let h = spec.hidden_size()
  let q = spec.q_lora_rank()
  let d = spec.head_dimension()
  let heads = spec.attention_head_count()
  let output_a = heads / spec.output_group_count() * spec.o_lora_rank()
  let output_b = spec.output_group_count() * spec.o_lora_rank()
  let f32 = Float32
  let bf16 = BFloat16
  match semantic.role {
    TokenEmbedding | OutputHead | MtpTokenEmbedding | MtpOutputHead =>
      Some((BF16, shape2(spec.vocabulary_size(), h), bf16))
    FinalNorm
    | AttentionNorm
    | FfnNorm
    | MtpEmbeddingNorm
    | MtpHiddenNorm
    | MtpFinalNorm
    | DsparkMainNorm => Some((BF16, shape1(h), bf16))
    HeadHyperConnection(HyperBase) => Some((F32, shape1(4), f32))
    HeadHyperConnection(HyperFunction) => Some((F32, shape2(4, 4 * h), f32))
    HeadHyperConnection(HyperScale) => Some((F32, shape1(1), f32))
    AttentionHyperConnection(HyperBase) | FfnHyperConnection(HyperBase) =>
      Some((F32, shape1(24), f32))
    AttentionHyperConnection(HyperFunction)
    | FfnHyperConnection(HyperFunction) => Some((F32, shape2(24, 4 * h), f32))
    AttentionHyperConnection(HyperScale) | FfnHyperConnection(HyperScale) =>
      Some((F32, shape1(3), f32))
    Attention(AttentionSink) => Some((F32, shape1(heads), f32))
    Attention(QueryNorm) => Some((BF16, shape1(q), bf16))
    Attention(KeyValueNorm) => Some((BF16, shape1(d), bf16))
    Attention(QueryA(part)) => Some(quant_layout(part, q, h))
    Attention(QueryB(part)) => Some(quant_layout(part, heads * d, q))
    Attention(KeyValue(part)) => Some(quant_layout(part, d, h))
    Attention(OutputA(part)) =>
      Some(quant_layout(part, output_a, heads * d / spec.output_group_count()))
    Attention(OutputB(part)) => Some(quant_layout(part, h, output_b))
    RouterWeight => Some((BF16, shape2(spec.routed_expert_count(), h), bf16))
    RouterBias => Some((F32, shape1(spec.routed_expert_count()), f32))
    TokenExpertHash =>
      Some(
        (I64, shape2(spec.vocabulary_size(), spec.experts_per_token()), Int64),
      )
    SharedExpert(matrix, part) => Some(expert_shape(spec, matrix, part, false))
    RoutedExpert(_, matrix, part) =>
      Some(expert_shape(spec, matrix, part, true))
    AttentionCompressor(role) => compressor_layout(spec, semantic.scope, role)
    AttentionIndexer(role) => indexer_layout(spec, semantic.scope, role)
    MtpEmbeddingProjection(part) | MtpHiddenProjection(part) =>
      Some(quant_layout(part, h, h))
    DsparkMainProjection(part) =>
      Some(quant_layout(part, h, h * spec.dspark_target_layer_ids().length()))
    DsparkConfidenceProjection =>
      Some((BF16, shape2(1, h + spec.dspark_markov_rank()), bf16))
    DsparkMarkovW1 | DsparkMarkovW2 =>
      Some(
        (BF16, shape2(spec.vocabulary_size(), spec.dspark_markov_rank()), bf16),
      )
  }
}

///|
fn quant_layout(
  part : DeepSeekV4QuantizedPart,
  rows : Int,
  columns : Int,
) -> (
  DeepSeekV4SafetensorsDType,
  ReadOnlyArray[Int64],
  DeepSeekV4WeightEncoding,
) {
  match part {
    QuantizedWeight => (F8E4M3, shape2(rows, columns), Fp8E4M3Block128)
    QuantizationScale => (F8E8M0, scale2(rows, columns), Ue8M0Scale)
  }
}

///|
fn compression_ratio(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  scope : DeepSeekV4TensorScope,
) -> Int? {
  match scope {
    DecoderLayer(index) => spec.compression_ratio(index)
    PredictionStage(index) => spec.compression_ratio(spec.layer_count() + index)
    Global => None
  }
}

///|
fn compressor_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  scope : DeepSeekV4TensorScope,
  role : DeepSeekV4CompressorTensor,
) -> (
  DeepSeekV4SafetensorsDType,
  ReadOnlyArray[Int64],
  DeepSeekV4WeightEncoding,
)? {
  guard compression_ratio(spec, scope) is Some(ratio) && ratio != 0 else {
    return None
  }
  let width = if ratio == 4 {
    2 * spec.head_dimension()
  } else {
    spec.head_dimension()
  }
  match role {
    CompressorApe => Some((F32, shape2(ratio, width), Float32))
    CompressorNorm => Some((BF16, shape1(spec.head_dimension()), BFloat16))
    CompressorGate | CompressorKeyValue =>
      Some((BF16, shape2(width, spec.hidden_size()), BFloat16))
  }
}

///|
fn indexer_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  scope : DeepSeekV4TensorScope,
  role : DeepSeekV4IndexerTensor,
) -> (
  DeepSeekV4SafetensorsDType,
  ReadOnlyArray[Int64],
  DeepSeekV4WeightEncoding,
)? {
  guard compression_ratio(spec, scope) is Some(4) else { return None }
  let j = spec.index_head_dimension()
  match role {
    IndexCompressorApe => Some((F32, shape2(4, 2 * j), Float32))
    IndexCompressorNorm => Some((BF16, shape1(j), BFloat16))
    IndexCompressorGate | IndexCompressorKeyValue =>
      Some((BF16, shape2(2 * j, spec.hidden_size()), BFloat16))
    IndexWeightsProjection =>
      Some(
        (BF16, shape2(spec.index_head_count(), spec.hidden_size()), BFloat16),
      )
    IndexQuery(part) =>
      Some(quant_layout(part, spec.index_head_count() * j, spec.q_lora_rank()))
  }
}

///|
fn is_common_block_role(role : DeepSeekV4TensorRole) -> Bool {
  match role {
    Attention(_)
    | AttentionNorm
    | RouterWeight
    | SharedExpert(_, _)
    | RoutedExpert(_, _, _)
    | FfnNorm
    | AttentionHyperConnection(_)
    | FfnHyperConnection(_) => true
    _ => false
  }
}

///|
fn valid_scope_role(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  semantic : DeepSeekV4TensorSemantic,
) -> Bool {
  match semantic.scope {
    Global =>
      match semantic.role {
        TokenEmbedding | OutputHead | FinalNorm | HeadHyperConnection(_) => true
        _ => false
      }
    DecoderLayer(layer) => {
      if is_common_block_role(semantic.role) {
        return true
      }
      match semantic.role {
        TokenExpertHash => layer < spec.hash_layer_count()
        RouterBias => layer >= spec.hash_layer_count()
        AttentionCompressor(_) =>
          compression_ratio(spec, semantic.scope).unwrap_or(0) != 0
        AttentionIndexer(_) =>
          compression_ratio(spec, semantic.scope) == Some(4)
        _ => false
      }
    }
    PredictionStage(stage) => {
      if is_common_block_role(semantic.role) {
        return true
      }
      if semantic.role is RouterBias {
        return true
      }
      if !spec.has_dspark() {
        match semantic.role {
          MtpTokenEmbedding | MtpOutputHead =>
            spec.profile() is (FlashBase | ProBase)
          MtpEmbeddingNorm
          | MtpHiddenNorm
          | MtpEmbeddingProjection(_)
          | MtpHiddenProjection(_)
          | MtpFinalNorm
          | HeadHyperConnection(_) => stage == 0
          _ => false
        }
      } else {
        match semantic.role {
          DsparkMainNorm | DsparkMainProjection(_) => stage == 0
          DsparkConfidenceProjection
          | DsparkMarkovW1
          | DsparkMarkovW2
          | MtpFinalNorm
          | HeadHyperConnection(_) => stage == 2
          _ => false
        }
      }
    }
  }
}

///|
pub fn expected_tensor_count(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> Int {
  let experts = 6 * spec.routed_expert_count()
  let mut total = 6
  for layer in 0..<spec.layer_count() {
    let non_expert = match spec.compression_ratio(layer).unwrap() {
      0 => 29
      4 => 40
      _ => 33
    }
    total += experts + non_expert
  }
  if spec.has_dspark() {
    total + (experts + 32) + (experts + 29) + (experts + 36)
  } else {
    let base_extra = if spec.profile() is (FlashBase | ProBase) { 2 } else { 0 }
    total + experts + 39 + base_extra
  }
}

///|
pub fn validate_tensor_semantic(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  tensor : DeepSeekV4SafetensorsTensor,
) -> DeepSeekV4TensorBinding raise DeepSeekV4TensorSchemaError {
  let semantic = match parse_semantic(tensor.name, spec) {
    Some(value) => value
    None => raise UnknownTensor(0)
  }
  if !valid_scope_role(spec, semantic) {
    raise UnknownTensor(0)
  }
  let (dtype, shape, encoding) = match expected_layout(spec, semantic) {
    Some(value) => value
    None => raise UnknownTensor(0)
  }
  if tensor.dtype != dtype || tensor.shape != shape {
    raise InvalidTensor(0)
  }
  { semantic, tensor, encoding, }
}

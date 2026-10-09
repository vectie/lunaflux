///|
/// Physical dtype tags admitted from official DeepSeek-V4 safetensors headers.
/// These are metadata semantics only and do not grant byte materialization.
pub(all) enum DeepSeekV4SafetensorsDType {
  BF16
  F32
  I64
  I8
  F8E4M3
  F8E8M0
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4TensorScope {
  Global
  DecoderLayer(Int)
  PredictionStage(Int)
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4QuantizedPart {
  QuantizedWeight
  QuantizationScale
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4ExpertMatrix {
  ExpertW1
  ExpertW2
  ExpertW3
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4HyperConnectionPart {
  HyperBase
  HyperFunction
  HyperScale
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4AttentionTensor {
  AttentionSink
  QueryNorm
  KeyValueNorm
  QueryA(DeepSeekV4QuantizedPart)
  QueryB(DeepSeekV4QuantizedPart)
  KeyValue(DeepSeekV4QuantizedPart)
  OutputA(DeepSeekV4QuantizedPart)
  OutputB(DeepSeekV4QuantizedPart)
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4CompressorTensor {
  CompressorApe
  CompressorNorm
  CompressorGate
  CompressorKeyValue
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4IndexerTensor {
  IndexCompressorApe
  IndexCompressorNorm
  IndexCompressorGate
  IndexCompressorKeyValue
  IndexWeightsProjection
  IndexQuery(DeepSeekV4QuantizedPart)
} derive(Debug, Eq)

///|
/// Closed semantic vocabulary for every tensor role in the five official
/// schema-v1 checkpoints.
pub(all) enum DeepSeekV4TensorRole {
  TokenEmbedding
  OutputHead
  FinalNorm
  HeadHyperConnection(DeepSeekV4HyperConnectionPart)
  Attention(DeepSeekV4AttentionTensor)
  AttentionCompressor(DeepSeekV4CompressorTensor)
  AttentionIndexer(DeepSeekV4IndexerTensor)
  AttentionNorm
  RouterWeight
  RouterBias
  TokenExpertHash
  SharedExpert(DeepSeekV4ExpertMatrix, DeepSeekV4QuantizedPart)
  RoutedExpert(Int, DeepSeekV4ExpertMatrix, DeepSeekV4QuantizedPart)
  FfnNorm
  AttentionHyperConnection(DeepSeekV4HyperConnectionPart)
  FfnHyperConnection(DeepSeekV4HyperConnectionPart)
  MtpTokenEmbedding
  MtpOutputHead
  MtpEmbeddingNorm
  MtpHiddenNorm
  MtpEmbeddingProjection(DeepSeekV4QuantizedPart)
  MtpHiddenProjection(DeepSeekV4QuantizedPart)
  MtpFinalNorm
  DsparkMainNorm
  DsparkMainProjection(DeepSeekV4QuantizedPart)
  DsparkConfidenceProjection
  DsparkMarkovW1
  DsparkMarkovW2
} derive(Debug, Eq)

///|
/// Fixed codebook semantics required to interpret packed official FP4 expert
/// weights. It is intentionally not represented as a fabricated tensor.
pub(all) enum DeepSeekV4Fp4Codebook {
  E2M1Canonical
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TensorSemantic {
  priv scope : DeepSeekV4TensorScope
  priv role : DeepSeekV4TensorRole
} derive(Debug, Eq)

///|
pub fn DeepSeekV4TensorSemantic::scope(self : Self) -> DeepSeekV4TensorScope {
  self.scope
}

///|
pub fn DeepSeekV4TensorSemantic::role(self : Self) -> DeepSeekV4TensorRole {
  self.role
}

///|
pub struct DeepSeekV4SafetensorsTensor {
  priv name : String
  priv dtype : DeepSeekV4SafetensorsDType
  priv shape : ReadOnlyArray[Int64]
  priv shard : Int
  priv data_start : Int64
  priv data_end : Int64
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TensorBinding {
  priv semantic : DeepSeekV4TensorSemantic
  priv tensor : DeepSeekV4SafetensorsTensor
  priv encoding : DeepSeekV4WeightEncoding
} derive(Debug, Eq)

///|
pub struct DeepSeekV4WeightBinding {
  priv profile : @deepseek_v4_spec.DeepSeekV4Profile
  priv identity : @spec.ModelIdentity
  priv tensors : ReadOnlyArray[DeepSeekV4TensorBinding]
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4MaterializationReadiness {
  UnsupportedByteLayouts(ReadOnlyArray[DeepSeekV4WeightEncoding])
} derive(Debug, Eq)

///|
pub(all) suberror DeepSeekV4TensorSchemaError {
  InvalidName
  InvalidDTypeTag
  InvalidShape
  InvalidShard
  InvalidByteRange
  ByteLengthMismatch
  UnknownTensor(Int)
  InvalidTensor(Int)
  DuplicateTensor(Int, Int)
  IncorrectTensorCount(expected~ : Int, actual~ : Int)
  IdentityMismatch
} derive(Debug, Eq)

///|
fn dtype_width(dtype : DeepSeekV4SafetensorsDType) -> Int64 {
  match dtype {
    BF16 => 2L
    F32 => 4L
    I64 => 8L
    I8 | F8E4M3 | F8E8M0 => 1L
  }
}

///|
fn parse_dtype(
  tag : StringView,
) -> DeepSeekV4SafetensorsDType raise DeepSeekV4TensorSchemaError {
  match tag {
    "BF16" => BF16
    "F32" => F32
    "I64" => I64
    "I8" => I8
    "F8_E4M3" => F8E4M3
    "F8_E8M0" => F8E8M0
    _ => raise InvalidDTypeTag
  }
}

///|
pub fn DeepSeekV4SafetensorsTensor::new(
  name~ : String,
  dtype~ : StringView,
  shape~ : ArrayView[Int64],
  shard~ : Int,
  data_start~ : Int64,
  data_end~ : Int64,
) -> Self raise DeepSeekV4TensorSchemaError {
  if name.is_empty() ||
    name.length() > 256 ||
    !name[:].all(c => c.is_ascii() && !c.is_ascii_control()) {
    raise InvalidName
  }
  if shard < 0 {
    raise InvalidShard
  }
  if shape.is_empty() || shape.length() > 4 {
    raise InvalidShape
  }
  let parsed_dtype = parse_dtype(dtype)
  let mut bytes = dtype_width(parsed_dtype)
  for dimension in shape {
    if dimension <= 0L || bytes > 9223372036854775807L / dimension {
      raise InvalidShape
    }
    bytes *= dimension
  }
  if data_start < 0L || data_end < data_start {
    raise InvalidByteRange
  }
  if data_end - data_start != bytes {
    raise ByteLengthMismatch
  }
  {
    name,
    dtype: parsed_dtype,
    shape: ReadOnlyArray::from_array(shape),
    shard,
    data_start,
    data_end,
  }
}

///|
pub fn DeepSeekV4SafetensorsTensor::name(self : Self) -> String {
  self.name
}

///|
pub fn DeepSeekV4SafetensorsTensor::dtype(
  self : Self,
) -> DeepSeekV4SafetensorsDType {
  self.dtype
}

///|
pub fn DeepSeekV4SafetensorsTensor::shape(self : Self) -> ReadOnlyArray[Int64] {
  self.shape
}

///|
pub fn DeepSeekV4SafetensorsTensor::shard(self : Self) -> Int {
  self.shard
}

///|
pub fn DeepSeekV4SafetensorsTensor::data_start(self : Self) -> Int64 {
  self.data_start
}

///|
pub fn DeepSeekV4SafetensorsTensor::data_end(self : Self) -> Int64 {
  self.data_end
}

///|
pub fn DeepSeekV4TensorBinding::semantic(
  self : Self,
) -> DeepSeekV4TensorSemantic {
  self.semantic
}

///|
pub fn DeepSeekV4TensorBinding::tensor(
  self : Self,
) -> DeepSeekV4SafetensorsTensor {
  self.tensor
}

///|
pub fn DeepSeekV4TensorBinding::encoding(
  self : Self,
) -> DeepSeekV4WeightEncoding {
  self.encoding
}

///|
pub fn DeepSeekV4WeightBinding::profile(
  self : Self,
) -> @deepseek_v4_spec.DeepSeekV4Profile {
  self.profile
}

///|
pub fn DeepSeekV4WeightBinding::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeepSeekV4WeightBinding::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn DeepSeekV4WeightBinding::tensor(
  self : Self,
  index : Int,
) -> DeepSeekV4TensorBinding? {
  self.tensors.get(index)
}

///|
pub fn DeepSeekV4WeightBinding::fp4_codebook(
  self : Self,
) -> DeepSeekV4Fp4Codebook? {
  if self.profile is (Flash | Pro | Flash0731) {
    Some(E2M1Canonical)
  } else {
    None
  }
}

///|
pub fn DeepSeekV4WeightBinding::materialization_readiness(
  self : Self,
) -> DeepSeekV4MaterializationReadiness {
  let model_id = match self.profile {
    Flash => "deepseek-ai/DeepSeek-V4-Flash"
    FlashBase => "deepseek-ai/DeepSeek-V4-Flash-Base"
    Pro => "deepseek-ai/DeepSeek-V4-Pro"
    ProBase => "deepseek-ai/DeepSeek-V4-Pro-Base"
    Flash0731 => "deepseek-ai/DeepSeek-V4-Flash-0731"
  }
  let spec = try! @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    model_id,
  )
  materialization_readiness(spec)
}

///|
/// Reports the current family loader boundary without implying that recognized
/// metadata can be decoded. This remains unsupported until every required
/// byte layout has an explicit bounded host/device materializer.
pub fn materialization_readiness(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4MaterializationReadiness {
  UnsupportedByteLayouts(requirements(spec).encodings())
}

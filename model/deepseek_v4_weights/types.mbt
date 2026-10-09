///|
/// Exact storage semantics present in official DeepSeek-V4 checkpoints.
/// Recognition grants no materialization or kernel authority.
pub(all) enum DeepSeekV4WeightEncoding {
  BFloat16
  Float32
  Int64
  Fp8E4M3Block128
  Ue8M0Scale
  PackedFp4E2M1Block32
} derive(Debug, Eq)

///|
/// Invalid construction of a DeepSeek-V4 weight backend declaration.
pub(all) suberror DeepSeekV4WeightsError {
  DuplicateEncoding(DeepSeekV4WeightEncoding)
} derive(Debug, Eq)

///|
/// Closed storage requirements derived from one exact semantic profile.
pub struct DeepSeekV4WeightRequirements {
  priv profile : @deepseek_v4_spec.DeepSeekV4Profile
  priv encodings : ReadOnlyArray[DeepSeekV4WeightEncoding]
} derive(Debug, Eq)

///|
/// Storage encodings implemented by one candidate loader/backend.
pub struct DeepSeekV4WeightBackendCapabilities {
  priv supported : FixedArray[Bool]
} derive(Debug, Eq)

///|
/// Startup join for the weight-storage boundary. Satisfying this contract is
/// necessary but is not tensor-vocabulary, content, or execution authority.
pub(all) enum DeepSeekV4WeightAdmission {
  StorageContractSatisfied(DeepSeekV4WeightRequirements)
  MissingWeightEncodings(
    DeepSeekV4WeightRequirements,
    ReadOnlyArray[DeepSeekV4WeightEncoding]
  )
} derive(Debug, Eq)

///|
fn DeepSeekV4WeightEncoding::stable_index(self : Self) -> Int {
  match self {
    BFloat16 => 0
    Float32 => 1
    Int64 => 2
    Fp8E4M3Block128 => 3
    Ue8M0Scale => 4
    PackedFp4E2M1Block32 => 5
  }
}

///|
pub fn requirements(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4WeightRequirements {
  let encodings : Array[DeepSeekV4WeightEncoding] = [
    BFloat16,
    Float32,
    Int64,
    Fp8E4M3Block128,
    Ue8M0Scale,
  ]
  if spec.expert_storage() is Fp4E2M1 {
    encodings.push(PackedFp4E2M1Block32)
  }
  { profile: spec.profile(), encodings: ReadOnlyArray::from_array(encodings), }
}

///|
pub fn DeepSeekV4WeightRequirements::profile(
  self : Self,
) -> @deepseek_v4_spec.DeepSeekV4Profile {
  self.profile
}

///|
pub fn DeepSeekV4WeightRequirements::encodings(
  self : Self,
) -> ReadOnlyArray[DeepSeekV4WeightEncoding] {
  self.encodings
}

///|
pub fn DeepSeekV4WeightBackendCapabilities::new(
  encodings : ArrayView[DeepSeekV4WeightEncoding],
) -> DeepSeekV4WeightBackendCapabilities raise DeepSeekV4WeightsError {
  let supported = FixedArray::make(6, false)
  for encoding in encodings {
    let index = encoding.stable_index()
    if supported[index] {
      raise DuplicateEncoding(encoding)
    }
    supported[index] = true
  }
  { supported, }
}

///|
pub fn DeepSeekV4WeightBackendCapabilities::supports(
  self : Self,
  encoding : DeepSeekV4WeightEncoding,
) -> Bool {
  self.supported[encoding.stable_index()]
}

///|
pub fn DeepSeekV4WeightRequirements::admit(
  self : Self,
  backend : DeepSeekV4WeightBackendCapabilities,
) -> DeepSeekV4WeightAdmission {
  let missing : Array[DeepSeekV4WeightEncoding] = []
  for encoding in self.encodings {
    if !backend.supports(encoding) {
      missing.push(encoding)
    }
  }
  if missing.is_empty() {
    StorageContractSatisfied(self)
  } else {
    MissingWeightEncodings(self, ReadOnlyArray::from_array(missing))
  }
}

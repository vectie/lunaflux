///|
fn DeepSeekV4NumericTensorRole::stable_index(self : Self) -> Int {
  match self {
    GlobalBFloat16Parameter => 0
    GlobalHyperConnectionFloat32 => 1
    AttentionBFloat16Parameter => 2
    AttentionFloat32Control => 3
    AttentionFp8Parameter => 4
    AttentionScaleMetadata => 5
    RouterBFloat16Weight => 6
    RouterFloat32Bias => 7
    TokenExpertHashInt64 => 8
    SharedExpertFp8Parameter => 9
    SharedExpertScaleMetadata => 10
    RoutedExpertFp8Parameter => 11
    RoutedExpertFp4Parameter => 12
    RoutedExpertBlock128ScaleMetadata => 13
    RoutedExpertBlock32ScaleMetadata => 14
    FfnNormBFloat16 => 15
    HyperConnectionFloat32 => 16
    CompressorBFloat16Parameter => 17
    CompressorFloat32Control => 18
    IndexerBFloat16Parameter => 19
    IndexerFloat32Control => 20
    IndexerFp8Parameter => 21
    IndexerScaleMetadata => 22
    MtpBFloat16Parameter => 23
    MtpHyperConnectionFloat32 => 24
    MtpFp8Parameter => 25
    MtpScaleMetadata => 26
    DsparkBFloat16Parameter => 27
    DsparkHyperConnectionFloat32 => 28
    DsparkFp8Parameter => 29
    DsparkScaleMetadata => 30
  }
}

///|
fn add_role(
  counts : FixedArray[Int],
  role : DeepSeekV4NumericTensorRole,
  count : Int,
) -> Unit {
  counts[role.stable_index()] += count
}

///|
fn add_block_roles(
  counts : FixedArray[Int],
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  compression_ratio : Int,
  hash_routed : Bool,
) -> Unit {
  add_role(counts, AttentionBFloat16Parameter, 3)
  add_role(counts, AttentionFloat32Control, 1)
  add_role(counts, AttentionFp8Parameter, 5)
  add_role(counts, AttentionScaleMetadata, 5)
  add_role(counts, RouterBFloat16Weight, 1)
  if hash_routed {
    add_role(counts, TokenExpertHashInt64, 1)
  } else {
    add_role(counts, RouterFloat32Bias, 1)
  }
  add_role(counts, SharedExpertFp8Parameter, 3)
  add_role(counts, SharedExpertScaleMetadata, 3)
  let routed_count = 3 * spec.routed_expert_count()
  if spec.expert_storage() is Fp4E2M1 {
    add_role(counts, RoutedExpertFp4Parameter, routed_count)
    add_role(counts, RoutedExpertBlock32ScaleMetadata, routed_count)
  } else {
    add_role(counts, RoutedExpertFp8Parameter, routed_count)
    add_role(counts, RoutedExpertBlock128ScaleMetadata, routed_count)
  }
  add_role(counts, FfnNormBFloat16, 1)
  add_role(counts, HyperConnectionFloat32, 6)
  if compression_ratio != 0 {
    add_role(counts, CompressorBFloat16Parameter, 3)
    add_role(counts, CompressorFloat32Control, 1)
  }
  if compression_ratio == 4 {
    add_role(counts, IndexerBFloat16Parameter, 4)
    add_role(counts, IndexerFloat32Control, 1)
    add_role(counts, IndexerFp8Parameter, 1)
    add_role(counts, IndexerScaleMetadata, 1)
  }
}

///|
fn role_for_index(index : Int) -> DeepSeekV4NumericTensorRole {
  match index {
    0 => GlobalBFloat16Parameter
    1 => GlobalHyperConnectionFloat32
    2 => AttentionBFloat16Parameter
    3 => AttentionFloat32Control
    4 => AttentionFp8Parameter
    5 => AttentionScaleMetadata
    6 => RouterBFloat16Weight
    7 => RouterFloat32Bias
    8 => TokenExpertHashInt64
    9 => SharedExpertFp8Parameter
    10 => SharedExpertScaleMetadata
    11 => RoutedExpertFp8Parameter
    12 => RoutedExpertFp4Parameter
    13 => RoutedExpertBlock128ScaleMetadata
    14 => RoutedExpertBlock32ScaleMetadata
    15 => FfnNormBFloat16
    16 => HyperConnectionFloat32
    17 => CompressorBFloat16Parameter
    18 => CompressorFloat32Control
    19 => IndexerBFloat16Parameter
    20 => IndexerFloat32Control
    21 => IndexerFp8Parameter
    22 => IndexerScaleMetadata
    23 => MtpBFloat16Parameter
    24 => MtpHyperConnectionFloat32
    25 => MtpFp8Parameter
    26 => MtpScaleMetadata
    27 => DsparkBFloat16Parameter
    28 => DsparkHyperConnectionFloat32
    29 => DsparkFp8Parameter
    30 => DsparkScaleMetadata
    _ => abort("DeepSeek numeric tensor role index escaped closed vocabulary")
  }
}

///|
fn storage_for_role(
  role : DeepSeekV4NumericTensorRole,
) -> DeepSeekV4LogicalStorage {
  match role {
    GlobalBFloat16Parameter
    | AttentionBFloat16Parameter
    | RouterBFloat16Weight
    | FfnNormBFloat16
    | CompressorBFloat16Parameter
    | IndexerBFloat16Parameter
    | MtpBFloat16Parameter
    | DsparkBFloat16Parameter => PlainBFloat16
    GlobalHyperConnectionFloat32
    | AttentionFloat32Control
    | RouterFloat32Bias
    | HyperConnectionFloat32
    | CompressorFloat32Control
    | IndexerFloat32Control
    | MtpHyperConnectionFloat32
    | DsparkHyperConnectionFloat32 => PlainFloat32
    TokenExpertHashInt64 => PlainInt64
    AttentionFp8Parameter
    | SharedExpertFp8Parameter
    | RoutedExpertFp8Parameter
    | IndexerFp8Parameter
    | MtpFp8Parameter
    | DsparkFp8Parameter => Fp8E4M3Block128Parameter
    AttentionScaleMetadata
    | SharedExpertScaleMetadata
    | RoutedExpertBlock128ScaleMetadata
    | IndexerScaleMetadata
    | MtpScaleMetadata
    | DsparkScaleMetadata => Ue8M0Block128ScaleMetadata
    RoutedExpertFp4Parameter => PackedFp4E2M1Block32Parameter
    RoutedExpertBlock32ScaleMetadata => Ue8M0Block32ScaleMetadata
  }
}

///|
fn build_tensor_runs(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> (Array[DeepSeekV4NumericTensorRun], Int) raise DeepSeekV4NumericPlanError {
  let counts = FixedArray::make(31, 0)
  add_role(counts, GlobalBFloat16Parameter, 3)
  add_role(counts, GlobalHyperConnectionFloat32, 3)
  for layer in 0..<spec.layer_count() {
    guard spec.compression_ratio(layer) is Some(ratio) else {
      raise ProfilePlanMismatch
    }
    add_block_roles(counts, spec, ratio, layer < spec.hash_layer_count())
  }
  for stage in 0..<spec.prediction_stage_count() {
    guard spec.compression_ratio(spec.layer_count() + stage) is Some(ratio) else {
      raise ProfilePlanMismatch
    }
    add_block_roles(counts, spec, ratio, false)
  }
  if spec.has_dspark() {
    add_role(counts, DsparkBFloat16Parameter, 5)
    add_role(counts, DsparkHyperConnectionFloat32, 3)
    add_role(counts, DsparkFp8Parameter, 1)
    add_role(counts, DsparkScaleMetadata, 1)
  } else {
    let base_embeddings = if spec.profile() is (FlashBase | ProBase) {
      2
    } else {
      0
    }
    add_role(counts, MtpBFloat16Parameter, 3 + base_embeddings)
    add_role(counts, MtpHyperConnectionFloat32, 3)
    add_role(counts, MtpFp8Parameter, 2)
    add_role(counts, MtpScaleMetadata, 2)
  }
  let runs : Array[DeepSeekV4NumericTensorRun] = []
  let mut total = 0
  for index in 0..<counts.length() {
    let count = counts[index]
    if count > 0 {
      let role = role_for_index(index)
      runs.push({ role, storage: storage_for_role(role), tensor_count: count, })
      total += count
    }
  }
  let expected = @deepseek_v4_weights.expected_tensor_count(spec)
  if total != expected || total > 145118 {
    raise ManifestCardinalityMismatch(expected~, actual=total)
  }
  (runs, total)
}

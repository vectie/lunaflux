///|
test "DeepSeek V4 instruct and base checkpoints retain distinct expert storage" {
  let instruct = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Pro",
  )
  let base = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Pro-Base",
  )
  let instruct_weights = @deepseek_v4_weights.requirements(instruct)
  let base_weights = @deepseek_v4_weights.requirements(base)
  assert_eq(instruct_weights.encodings(), [
    BFloat16,
    Float32,
    Int64,
    Fp8E4M3Block128,
    Ue8M0Scale,
    PackedFp4E2M1Block32,
  ])
  assert_eq(base_weights.encodings(), [
    BFloat16,
    Float32,
    Int64,
    Fp8E4M3Block128,
    Ue8M0Scale,
  ])
}

///|
test "DeepSeek V4 FP4 checkpoints fail closed on the current numeric subset" {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let required = @deepseek_v4_weights.requirements(spec)
  let current_subset = @deepseek_v4_weights.DeepSeekV4WeightBackendCapabilities::new([
      BFloat16,
      Float32,
      Int64,
      Fp8E4M3Block128,
      Ue8M0Scale,
    ],
  )
  match required.admit(current_subset) {
    MissingWeightEncodings(profile, missing) => {
      assert_eq(profile.profile(), Flash0731)
      assert_eq(missing, [PackedFp4E2M1Block32])
    }
    StorageContractSatisfied(_) =>
      fail("backend without packed FP4 admitted an instruct checkpoint")
  }
}

///|
test "DeepSeek V4 base storage contract admits an exact complete backend" {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-Base",
  )
  let required = @deepseek_v4_weights.requirements(spec)
  let backend = @deepseek_v4_weights.DeepSeekV4WeightBackendCapabilities::new([
    BFloat16,
    Float32,
    Int64,
    Fp8E4M3Block128,
    Ue8M0Scale,
  ])
  match required.admit(backend) {
    StorageContractSatisfied(profile) => assert_eq(profile, required)
    MissingWeightEncodings(_, _) => fail("complete base storage was rejected")
  }
}

///|
test "DeepSeek V4 weight capability declarations reject duplicates" {
  try
    @deepseek_v4_weights.DeepSeekV4WeightBackendCapabilities::new([
      BFloat16,
      BFloat16,
    ])
  catch {
    DuplicateEncoding(BFloat16) => ()
    DuplicateEncoding(_) => fail("unexpected duplicate encoding rejection")
  } noraise {
    _ => fail("duplicate weight encoding was admitted")
  }
}

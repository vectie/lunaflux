///|
test "family requirements admit only a complete backend" {
  let requirements = try! @family_contract.ModelFamilyRequirements::new(
    family=DeepSeekV4,
    architecture="DeepseekV4ForCausalLM",
    model_type="deepseek_v4",
    schema_version=1,
    workload=DecoderOnlyText,
    capabilities=[
      DenseCausalAttention,
      MixtureOfExperts,
      HyperConnections,
      CompressedSparseAttention,
    ],
  )
  let partial = try! @family_contract.ModelBackendCapabilities::new([
    DenseCausalAttention,
  ])
  match requirements.admit(partial) {
    MissingCapabilities(profile, missing) => {
      assert_eq(profile.family(), DeepSeekV4)
      assert_eq(missing.length(), 3)
      assert_eq(missing[0], MixtureOfExperts)
      assert_eq(missing[1], HyperConnections)
      assert_eq(missing[2], CompressedSparseAttention)
    }
    SemanticCapabilitiesAvailable(_) =>
      fail("partial backend admitted DeepSeek V4")
  }
  let complete = try! @family_contract.ModelBackendCapabilities::new([
    DenseCausalAttention,
    MixtureOfExperts,
    HyperConnections,
    CompressedSparseAttention,
  ])
  assert_true(requirements.admit(complete) is SemanticCapabilitiesAvailable(_))
}

///|
test "family contracts reject ambiguous capability declarations" {
  try
    @family_contract.ModelBackendCapabilities::new([
      DenseCausalAttention,
      DenseCausalAttention,
    ])
  catch {
    DuplicateCapability(DenseCausalAttention) => ()
    _ => fail("unexpected duplicate capability error")
  } noraise {
    _ => fail("duplicate backend capability was admitted")
  }
  try
    @family_contract.ModelFamilyRequirements::new(
      family=MiniMaxH3,
      architecture="MiniMaxH3DiTModel",
      model_type="minimax_h3",
      schema_version=0,
      workload=AudioVideoDiffusion,
      capabilities=[DiffusionDenoising],
    )
  catch {
    InvalidSchemaVersion => ()
    _ => fail("unexpected schema-version error")
  } noraise {
    _ => fail("invalid family schema version was admitted")
  }
}

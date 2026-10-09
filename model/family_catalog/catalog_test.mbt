///|
test "catalog keeps family variants modular and fail closed" {
  let profiles = @family_catalog.profiles()
  assert_eq(profiles.length(), 4)
  assert_eq(profiles[0].family(), DeepSeekV4)
  assert_eq(profiles[0].architecture(), "DeepseekV4ForCausalLM")
  assert_eq(profiles[1].family(), MiniMaxH3)
  assert_eq(profiles[1].workload(), AudioVideoDiffusion)
  assert_eq(profiles[2].family(), Glm53)
  assert_eq(profiles[2].architecture(), "GlmMoeDsaForCausalLM")
  assert_eq(profiles[2].workload(), DecoderOnlyText)
  assert_eq(profiles[3].family(), Glm53)
  assert_eq(profiles[3].architecture(), "Glm5NextForConditionalGeneration")
  assert_eq(profiles[3].workload(), MultimodalConditionalGeneration)

  let admissions = @family_catalog.admit_profiles(
    @family_catalog.baseline_dense_decoder_backend(),
  )
  assert_eq(admissions.length(), 4)
  for admission in admissions {
    match admission {
      MissingCapabilities(_, missing) => assert_true(!missing.is_empty())
      SemanticCapabilitiesAvailable(_) =>
        fail("advanced family was admitted by dense backend")
    }
  }
}

///|
test "catalog can admit one exact complete family backend" {
  let deepseek = @family_catalog.profiles()[0]
  let backend = try! @family_contract.ModelBackendCapabilities::new(
    deepseek.capabilities(),
  )
  assert_true(deepseek.admit(backend) is SemanticCapabilitiesAvailable(_))
  let admissions = @family_catalog.admit_profiles(backend)
  assert_true(admissions[0] is SemanticCapabilitiesAvailable(_))
  for index in 1..<admissions.length() {
    assert_true(admissions[index] is MissingCapabilities(_, _))
  }
}

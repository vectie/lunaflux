///|
fn responses_comparison_test_matrix(
  changed_identity_index? : Int = -1,
) -> Array[OpenAIResponsesTrialEvidence] raise {
  let trials : Array[OpenAIResponsesTrialEvidence] = []
  let engines = openai_comparison_engines()
  for profile in openai_comparison_profiles() {
    for ordinal in 1..<=3 {
      for engine_index, engine in engines {
        let index = trials.length()
        let identity = if index == changed_identity_index {
          OpenAIBaselineIdentity::new(
            engine,
            baseline_test_digest('1'),
            baseline_test_digest('9'),
            baseline_test_digest('3'),
          )
        } else {
          baseline_test_identity(engine~)
        }
        let observer = OpenAIResponsesTrialContract::new(
          identity,
          baseline_test_inputs(profile~),
          ordinal,
          comparison_test_position(engine_index, ordinal),
          2048,
          16,
        ).start()
        trials.push(responses_test_complete(observer))
      }
    }
  }
  trials
}

///|
test "Responses measurement bridge admits exact inert 81-trial matrix" {
  let admitted = admit_openai_responses_comparison_measurements(
    comparison_test_pins(),
    responses_comparison_test_matrix(),
  )
  admitted.require_responses_v1_observation()
  assert_eq(admitted.trial_count(), 81)
  assert_eq(admitted.engine_identity(0).engine(), LunaFlux)
  assert_eq(admitted.engine_identity(1).engine(), Vllm)
  assert_eq(admitted.engine_identity(2).engine(), Sglang)
  assert_eq(admitted.measurements(0).summary().completed(), 1)
}

///|
test "Responses measurement bridge rejects one image-pin substitution" {
  try
    admit_openai_responses_comparison_measurements(
      comparison_test_pins(),
      responses_comparison_test_matrix(changed_identity_index=80),
    )
  catch {
    OpenAIBaselineInputMismatch => ()
    _ => fail("unexpected Responses identity mismatch rejection")
  } noraise {
    _ => fail("mixed Responses image identities were admitted")
  }
}

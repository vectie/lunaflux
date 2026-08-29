///|
fn comparison_test_position(engine : Int, ordinal : Int) -> Int {
  match ordinal {
    1 => engine
    2 => (engine + 2) % 3
    _ => (engine + 1) % 3
  }
}

///|
fn comparison_test_trial(
  engine : @evidence.BenchmarkEngine,
  profile : @evidence.BenchmarkProfile,
  ordinal : Int,
  order_position : Int,
  protocol_unit? : Char = '6',
) -> OpenAIBaselineTrialEvidence raise {
  let inputs = baseline_test_inputs(profile~, protocol_unit~)
  let adapter = OpenAIBaselineTrialContract::new(
    baseline_test_identity(engine~),
    inputs,
    ordinal,
    order_position,
    2048,
    16,
  ).start()
  drive_fake_server_completion(adapter)
}

///|
fn comparison_test_matrix(
  hostile_protocol_index? : Int = -1,
) -> Array[OpenAIBaselineTrialEvidence] raise {
  let trials : Array[OpenAIBaselineTrialEvidence] = []
  let engines = openai_comparison_engines()
  for profile in openai_comparison_profiles() {
    for ordinal in 1..<=3 {
      for engine_index, engine in engines {
        let index = trials.length()
        trials.push(
          comparison_test_trial(
            engine,
            profile,
            ordinal,
            comparison_test_position(engine_index, ordinal),
            protocol_unit=if index == hostile_protocol_index {
              '9'
            } else {
              '6'
            },
          ),
        )
      }
    }
  }
  trials
}

///|
fn comparison_test_pins() -> OpenAIComparisonEnvironmentPins raise {
  OpenAIComparisonEnvironmentPins::new(
    baseline_test_digest('a'),
    baseline_test_digest('b'),
    baseline_test_digest('c'),
    baseline_test_digest('d'),
    baseline_test_digest('e'),
  )
}

///|
test "branded OpenAI measurement set has exact three by nine by three shape" {
  let admitted = admit_openai_comparison_measurements(
    comparison_test_pins(),
    comparison_test_matrix(),
  )
  assert_eq(admitted.trial_count(), 81)
  assert_eq(admitted.engine_identity(0).engine(), LunaFlux)
  assert_eq(admitted.engine_identity(1).engine(), Vllm)
  assert_eq(admitted.engine_identity(2).engine(), Sglang)
  for index, profile in openai_comparison_profiles() {
    assert_eq(admitted.workload(index).profile(), profile)
  }
  assert_eq(admitted.measurements(0).summary().completed(), 1)
}

///|
test "comparison measurement set rejects one protocol substitution" {
  try
    admit_openai_comparison_measurements(
      comparison_test_pins(),
      comparison_test_matrix(hostile_protocol_index=80),
    )
  catch {
    OpenAIBaselineInputMismatch => ()
    _ => fail("unexpected mixed-protocol comparison rejection")
  } noraise {
    _ => fail("mixed native or OpenAI protocol measurements were admitted")
  }
}

///|
test "Chat measurement set refuses Responses approval" {
  let admitted = admit_openai_comparison_measurements(
    comparison_test_pins(),
    comparison_test_matrix(),
  )
  try admitted.require_responses_v1_observation() catch {
    OpenAIBaselineProtocolIncompatible => ()
    _ => fail("unexpected Responses incompatibility")
  } noraise {
    _ => fail("Chat Completions measurements satisfied Responses approval")
  }
}

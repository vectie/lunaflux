///|
test "FL2VA keyframe path matches independently calculated attention and Euler fixture" {
  let plan = tiny_transformer_plan()
  let result = @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
    plan,
    tiny_request(plan),
    conditioning(false),
    tiny_weights(plan),
    seed=42L,
    step_index=0,
    limits=limits(),
  )
  assert_eq(result.packed_projected().length(), 28)
  assert_eq(result.video_prediction().length(), 24)
  assert_eq(result.audio_prediction().length(), 4)
  let video_expected = [10.0 / 7.0, 3.0 / 7.0, 0.0, 10.0 / 7.0]
  for index in 0..<8 {
    assert_true(
      approximately(result.video_prediction()[index], video_expected[index % 4]),
    )
  }
  assert_true(approximately(result.audio_prediction()[0], 10.0 / 7.0))
  assert_true(approximately(result.audio_prediction()[1], 17.0 / 7.0))
  assert_true(approximately(result.video_sigma(), 1.0))
  assert_true(approximately(result.video_next_sigma(), 1.5005 / 1.75025))
  assert_true(approximately(result.audio_sigma(), 1.0))
  assert_true(approximately(result.audio_next_sigma(), 0.75025))
  for index in 0..<result.updated_video().length() {
    let expected = result.initial_video()[index] +
      (1.0 - result.video_next_sigma()) * result.video_prediction()[index]
    assert_true(approximately(result.updated_video()[index], expected))
  }
}

///|
test "both modalities use clean-state velocity sign at every step including terminal" {
  for reference in [false, true] {
    let plan = tiny_transformer_plan(reference~)
    let request = tiny_request(plan)
    for step_index in 0..<request.model_evaluations() {
      let result = @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
        plan,
        request,
        conditioning(reference),
        tiny_weights(plan),
        seed=42L,
        step_index~,
        limits=limits(),
      )
      for
        (initial, velocity, updated, sigma, next_sigma) in [
          (
            result.initial_video(),
            result.video_prediction(),
            result.updated_video(),
            result.video_sigma(),
            result.video_next_sigma(),
          ),
          (
            result.initial_audio(),
            result.audio_prediction(),
            result.updated_audio(),
            result.audio_sigma(),
            result.audio_next_sigma(),
          ),
        ] {
        for i in 0..<initial.length() {
          // Independently simplified Euler form rather than implementation's
          // clean-state blend; detects the old reversed velocity sign.
          let expected = initial[i] + (sigma - next_sigma) * velocity[i]
          assert_true(approximately(updated[i], expected))
          if velocity[i] > 0.0 {
            assert_true(updated[i] > initial[i])
          }
          if step_index == request.model_evaluations() - 1 {
            assert_eq(next_sigma, 0.0)
            assert_true(
              approximately(updated[i], initial[i] + sigma * velocity[i]),
            )
          }
        }
      }
    }
  }
}

///|
test "Ref2VA reference projection changes packed attention and modality MLP outputs" {
  let plan = tiny_transformer_plan(reference=true)
  let result = @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
    plan,
    tiny_request(plan),
    conditioning(true),
    tiny_weights(plan),
    seed=42L,
    step_index=0,
    limits=limits(),
  )
  let video_expected = [10.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 10.0 / 7.0]
  for index in 0..<8 {
    assert_true(
      approximately(result.video_prediction()[index], video_expected[index % 4]),
    )
  }
  assert_true(approximately(result.audio_prediction()[0], 10.0 / 7.0))
  assert_true(approximately(result.audio_prediction()[1], 16.0 / 7.0))
  assert_true(result.initial_video() != result.updated_video())
  assert_true(result.initial_audio() != result.updated_audio())
}

///|
test "official MiniMax variants expose only the tested generic conditioning stages" {
  fn official(
    variant : @minimax_h3_spec.MiniMaxH3Variant,
  ) -> @joint_diffusion_plan.JointDiffusionPlan {
    try! @minimax_h3.build(
      @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
        @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
          variant~,
          checkpoint_format=DiffusersV1,
        ),
        test_content('5'),
      ),
    )
  }
  assert_true(
    official(Fl2va)
    .stages()
    .any(stage => stage.kind() == EncodeKeyframeConditioning),
  )
  assert_true(
    official(Ref2va)
    .stages()
    .any(stage => stage.kind() == EncodeReferenceConditioning),
  )
}

///|
test "shape finite step and allocation bounds fail closed" {
  try
    @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=TextConditioning,
      rows=1,
      width=4,
      values=[0.0, 0.0, 0.0, 1.0 / 0.0],
    )
  catch {
    NonFiniteInput => ()
    _ => fail("unexpected nonfinite error")
  } noraise {
    _ => fail("nonfinite conditioning admitted")
  }
  let plan = tiny_transformer_plan()
  let both = try! @joint_diffusion_transformer_reference.JointTinyConditioningInput::new(
    text=try! @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=TextConditioning,
      rows=1,
      width=4,
      values=[0.0, 0.0, 0.0, 0.0],
    ),
    keyframe=try! @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=KeyframeConditioning,
      rows=1,
      width=2,
      values=[0.0, 0.0],
    ),
    reference=try! @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=ReferenceConditioning,
      rows=1,
      width=3,
      values=[0.0, 0.0, 0.0],
    ),
  )
  try
    @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      both,
      tiny_weights(plan),
      seed=42L,
      step_index=0,
      limits=limits(),
    )
  catch {
    InvalidConditioningContract => ()
    _ => fail("unexpected mixed conditioning error")
  } noraise {
    _ => fail("keyframe and reference conditioning were mixed")
  }
  try
    @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      conditioning(false),
      tiny_weights(plan),
      seed=42L,
      step_index=5,
      limits=limits(),
    )
  catch {
    InvalidStepIndex(5) => ()
    _ => fail("unexpected step error")
  } noraise {
    _ => fail("terminal step admitted")
  }
  try
    @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      conditioning(false),
      tiny_weights(plan),
      seed=42L,
      step_index=0,
      limits=limits(maximum_rows=6),
    )
  catch {
    InvalidPlan => ()
    _ => fail("unexpected row bound error")
  } noraise {
    _ => fail("row bound exceeded")
  }
  try
    @joint_diffusion_transformer_reference.run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      conditioning(false),
      tiny_weights(plan),
      seed=42L,
      step_index=0,
      limits=limits(maximum_elements=27),
    )
  catch {
    ElementBudgetExceeded => ()
    _ => fail("unexpected element bound error")
  } noraise {
    _ => fail("element budget exceeded")
  }
}

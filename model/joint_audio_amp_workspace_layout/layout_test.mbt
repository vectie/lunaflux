///|
test "first-stage AMP workspace binds 30+33+33+1 stages and disjoint live storage" {
  let layout = @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
    element_count=1059840L,
    intermediate_element_count=2119680L,
    normalized_weight_element_count=2883584L,
  )
  assert_eq(layout.stages().length(), 97)
  assert_eq(layout.regions().length(), 13)
  assert_eq(layout.workspace_bytes(), 58167296L)
  match layout.stages()[0] {
    Convolution(0, 0, 0, Normalize) => ()
    _ => fail("stage zero must consume the precomputed activation")
  }
  match layout.stages()[29] {
    Residual(0, 2) => ()
    _ => fail("block zero must contain 30 launches")
  }
  match layout.stages()[62] {
    Residual(1, 2) => ()
    _ => fail("block one must contain 33 launches")
  }
  match layout.stages()[95] {
    Residual(2, 2) => ()
    _ => fail("block two must contain 33 launches")
  }
  match layout.stages()[96] {
    ParallelAverage => ()
    _ => fail("the final launch must average the three blocks")
  }
}

///|
test "recomputed activation0 and 100-stage plans fail closed" {
  let layout = @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
    element_count=1059840L,
    intermediate_element_count=2119680L,
    normalized_weight_element_count=2883584L,
  )
  let stages : Array[
    @joint_audio_amp_workspace_layout.JointAudioAmpExecutionStage,
  ] = [
    Activation(0, 0, Upsample),
    Activation(0, 0, SnakeBeta),
    Activation(0, 0, Downsample),
  ]
  for stage in layout.stages() {
    stages.push(stage)
  }
  assert_eq(stages.length(), 100)
  try
    @joint_audio_amp_workspace_layout.validate_first_stage_amp_execution_stages(
      stages,
    )
  catch {
    InvalidStageCount => ()
    _ => fail("wrong 100-stage rejection")
  } noraise {
    _ => fail("100-stage plan admitted")
  }
}

///|
test "workspace geometry and overflow fail closed" {
  try
    @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
      element_count=9L,
      intermediate_element_count=9L,
      normalized_weight_element_count=1L,
    )
  catch {
    InvalidGeometry => ()
    _ => fail("wrong geometry rejection")
  } noraise {
    _ => fail("invalid geometry admitted")
  }
  try
    @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
      element_count=4611686018427387904L,
      intermediate_element_count=9223372036854775807L,
      normalized_weight_element_count=1L,
    )
  catch {
    InvalidGeometry => ()
    _ => fail("wrong overflow rejection")
  } noraise {
    _ => fail("overflowing geometry admitted")
  }
}

///|
test "second AMP stage reuses exact liveness with smaller authenticated geometry" {
  let layout = @joint_audio_amp_workspace_layout.amp_workspace_layout(
    element_count=2649600L,
    intermediate_element_count=5299200L,
    normalized_weight_element_count=720896L,
  )
  assert_eq(layout.stages().length(), 97)
  assert_eq(layout.workspace_bytes(), 119465984L)
  try! @joint_audio_amp_workspace_layout.validate_amp_execution_stages(
    layout.stages(),
  )
  inspect(
    layout.digest().as_hex(),
    content=(
      #|9d5fc9758af04043bef62b6591ea8fc3c78505e4accb1ebeea0cce36edbd6436
    ),
  )
}

///|
suberror Injected {
  Run
  Drain
  Release
}

///|
struct Frame {
  index : Int
  step : @joint_diffusion_loop.JointStep
  state : FixedArray[Float]
  events : Array[String]
  fail_run : Bool
  mut fail_drain : Bool
  mut fail_release : Bool
}

///|
impl @joint_diffusion_execution.PreparedFrame for Frame with fn execute(self) {
  self.events.push("execute\{self.index}")
  let video_velocity = self.state[0] + self.state[1]
  let audio_velocity = self.state[0] - self.state[1]
  self.state[0] = self.step.update_video(self.state[0], video_velocity)
  if self.fail_run {
    raise Run
  }
  self.state[1] = self.step.update_audio(self.state[1], audio_velocity)
}

///|
impl @joint_diffusion_execution.PreparedFrame for Frame with fn drain(self) {
  self.events.push("drain\{self.index}")
  if self.fail_drain {
    self.fail_drain = false
    raise Drain
  }
}

///|
impl @joint_diffusion_execution.PreparedFrame for Frame with fn release(self) {
  self.events.push("release\{self.index}")
  if self.fail_release {
    self.fail_release = false
    raise Release
  }
}

///|
fn plan() -> @joint_diffusion_loop.PreparedPlan raise {
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant=Fl2va,
        checkpoint_format=DiffusersV1,
      ),
      @spec.ContentDigest::from_sha256(String::make(64, '7')),
    ),
  )
  let request = model
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=2,
    )
  @joint_diffusion_loop.PreparedPlan::new(
    model,
    request,
    max_latent_elements=4000000L,
  )
}

///|
fn frames(
  plan : @joint_diffusion_loop.PreparedPlan,
  fail_run? : Int = -1,
) -> Array[Frame] {
  let state : FixedArray[Float] = [2.0, 1.0]
  let events : Array[String] = []
  [
    for index, step in plan.steps() => {
      {
        index,
        step,
        state,
        events,
        fail_run: index == fail_run,
        fail_drain: false,
        fail_release: false,
      }
    }
  ]
}

///|
test "prepared frames execute once in order and release leases after completion" {
  let plan = plan()
  let frames = frames(plan)
  let events = frames[0].events
  let expected : FixedArray[Float] = [2.0, 1.0]
  for step in plan.steps() {
    let v = expected[0] + expected[1]
    let a = expected[0] - expected[1]
    expected[0] = step.update_video(expected[0], v)
    expected[1] = step.update_audio(expected[1], a)
  }
  let execution = @joint_diffusion_execution.Execution::new(plan, frames)
  let state = frames[0].state
  // Only handle-array ownership is copied; mutating it cannot reorder a run.
  frames[0] = frames[1]
  assert_eq(execution.advance(), Running)
  assert_eq(execution.advance(), Completed)
  assert_eq(execution.completed_steps(), 2)
  assert_eq(state[0], expected[0])
  assert_eq(state[1], expected[1])
  execution.close()
  execution.close()
  assert_true(execution.is_closed())
  assert_eq(execution.progress(), Completed)
  assert_eq(events, [
    "execute0", "execute1", "drain0", "release0", "drain1", "release1",
  ])
  try execution.advance() catch {
    @joint_diffusion_execution.ExecutionError::Closing => ()
    _ => fail("closed owner admitted submission")
  } noraise {
    _ => fail("closed owner admitted submission")
  }
}

///|
test "cancellation drains and releases unsubmitted frames without executing them" {
  let plan = plan()
  let frames = frames(plan)
  let execution = @joint_diffusion_execution.Execution::new(plan, frames)
  ignore(execution.advance())
  execution.cancel()
  try execution.advance() catch {
    @joint_diffusion_loop.LoopError::NotRunning => ()
    _ => fail("wrong cancelled status")
  } noraise {
    _ => fail("cancelled frame executed")
  }
  execution.close()
  assert_eq(execution.completed_steps(), 1)
  assert_eq(execution.progress(), Cancelled)
  assert_eq(frames[0].events, [
    "execute0", "drain0", "release0", "drain1", "release1",
  ])
}

///|
test "partial execution failure poisons progress and cleanup failures remain retryable" {
  let plan = plan()
  let frames = frames(plan, fail_run=1)
  let execution = @joint_diffusion_execution.Execution::new(plan, frames)
  ignore(execution.advance())
  try execution.advance() catch {
    Run => ()
    _ => fail("unexpected run error")
  } noraise {
    _ => fail("missing run failure")
  }
  assert_eq(execution.progress(), Failed)
  assert_eq(execution.completed_steps(), 1)
  frames[1].fail_drain = true
  try execution.close() catch {
    Drain => ()
    _ => fail("unexpected drain error")
  } noraise {
    _ => fail("missing drain failure")
  }
  assert_false(execution.is_closed())
  frames[1].fail_release = true
  try execution.close() catch {
    Release => ()
    _ => fail("unexpected release error")
  } noraise {
    _ => fail("missing release failure")
  }
  assert_false(execution.is_closed())
  execution.close()
  assert_true(execution.is_closed())
  assert_eq(execution.progress(), Failed)
  // Frame 0 is never released twice; retry of release 1 does not drain again.
  assert_eq(frames[0].events, [
    "execute0", "execute1", "drain0", "release0", "drain1", "drain1", "release1",
    "release1",
  ])
}

///|
test "wrong frame count transfers no ownership" {
  let plan = plan()
  let frames = frames(plan)
  try @joint_diffusion_execution.Execution::new(plan, frames[:1]) catch {
    @joint_diffusion_execution.ExecutionError::InvalidFrameCount => ()
    _ => fail("wrong constructor failure")
  } noraise {
    _ => fail("wrong frame count accepted")
  }
  assert_eq(frames[0].events.length(), 0)
}

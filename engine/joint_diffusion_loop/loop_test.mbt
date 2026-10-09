///|
struct NumericBackend {
  video : FixedArray[Float]
  audio : FixedArray[Float]
  mut calls : Int
  fail_at : Int
}

///|
suberror TestFailure {
  Injected
}

///|
impl @joint_diffusion_loop.JointBackend for NumericBackend with fn execute_step(
  self,
  index,
  step,
) {
  assert_eq(index, self.calls)
  self.calls += 1
  // Joint prediction must consume both old states before either update.
  let video_prediction = self.video[0] + self.audio[0]
  let audio_prediction = self.video[0] - self.audio[0]
  self.video[0] = step.update_video(self.video[0], video_prediction)
  if index == self.fail_at {
    raise Injected
  }
  self.audio[0] = step.update_audio(self.audio[0], audio_prediction)
}

///|
fn backend(fail_at? : Int = -1) -> NumericBackend {
  {
    video: FixedArray::make(1, 2.0),
    audio: FixedArray::make(1, 1.0),
    calls: 0,
    fail_at,
  }
}

///|
fn steps() -> Array[@joint_diffusion_loop.JointStep] {
  [
    {
      video_timestep: 0.0,
      audio_timestep: 0.0,
      video_sigma: 1.0,
      audio_sigma: 1.0,
      video_next_sigma: 0.75,
      audio_next_sigma: 0.5,
    },
    {
      video_timestep: 0.25,
      audio_timestep: 0.5,
      video_sigma: 0.75,
      audio_sigma: 0.5,
      video_next_sigma: 0.0,
      audio_next_sigma: 0.0,
    },
  ]
}

///|
test "joint updates use old state and complete exactly once" {
  let driver = @joint_diffusion_loop.Loop::new(steps())
  let backend = backend()
  assert_eq(driver.advance(backend), Running)
  assert_eq(backend.video[0], 2.75)
  assert_eq(backend.audio[0], 1.5)
  assert_eq(driver.advance(backend), Completed)
  assert_eq(backend.video[0], 5.9375)
  assert_eq(backend.audio[0], 2.125)
  assert_eq(driver.completed_steps(), 2)
  driver.cancel()
  assert_eq(driver.progress(), Completed)
  try driver.advance(backend) catch {
    @joint_diffusion_loop.LoopError::NotRunning => ()
    _ => fail("wrong terminal error")
  } noraise {
    _ => fail("replayed terminal work")
  }
  assert_eq(backend.calls, 2)
}

///|
test "partial backend failure cannot replay or complete" {
  let driver = @joint_diffusion_loop.Loop::new(steps())
  let backend = backend(fail_at=0)
  try driver.advance(backend) catch {
    Injected => ()
    _ => fail("backend error lost")
  } noraise {
    _ => fail("failure swallowed")
  }
  assert_eq(driver.progress(), Failed)
  assert_eq(driver.completed_steps(), 0)
  assert_eq(backend.video[0], 2.75)
  assert_eq(backend.audio[0], 1.0)
  driver.cancel()
  assert_eq(driver.progress(), Failed)
  try driver.advance(backend) catch {
    @joint_diffusion_loop.LoopError::NotRunning => ()
    _ => fail("wrong failed-loop error")
  } noraise {
    _ => fail("partial step replayed")
  }
  assert_eq(backend.calls, 1)
}

///|
test "cancellation prevents next submission" {
  let driver = @joint_diffusion_loop.Loop::new(steps())
  let backend = backend()
  ignore(driver.advance(backend))
  driver.cancel()
  driver.cancel()
  assert_eq(driver.progress(), Cancelled)
  assert_eq(driver.completed_steps(), 1)
  try driver.advance(backend) catch {
    @joint_diffusion_loop.LoopError::NotRunning => ()
    _ => fail("wrong cancelled-loop error")
  } noraise {
    _ => fail("cancelled work submitted")
  }
  assert_eq(backend.calls, 1)
}

///|
test "H3 five-step dual shifted grid drives joint numeric backend" {
  let plan = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant=Fl2va,
        checkpoint_format=DiffusersV1,
      ),
      @spec.ContentDigest::from_sha256(String::make(64, '9')),
    ),
  )
  let request = plan
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
  let video = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
    plan,
    request,
    VideoLatent,
  )
  let audio = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
    plan,
    request,
    AudioLatent,
  )
  let steps : Array[@joint_diffusion_loop.JointStep] = [
    for i in 0..<5 => {
      {
        video_timestep: 1.0 - video.sigma(i).unwrap(),
        audio_timestep: 1.0 - audio.sigma(i).unwrap(),
        video_sigma: video.sigma(i).unwrap(),
        audio_sigma: audio.sigma(i).unwrap(),
        video_next_sigma: video.sigma(i + 1).unwrap(),
        audio_next_sigma: audio.sigma(i + 1).unwrap(),
      }
    }
  ]
  let driver = @joint_diffusion_loop.Loop::new(steps)
  let backend = backend()
  for i in 0..<5 {
    ignore(driver.advance(backend))
    assert_eq(driver.completed_steps(), i + 1)
  }
  assert_eq(driver.progress(), Completed)
  assert_eq(backend.calls, 5)
  assert_true(!backend.video[0].is_nan() && !backend.audio[0].is_nan())
}

///|
test "schedule copied at preparation and malformed input rejected" {
  let input = steps()
  let driver = @joint_diffusion_loop.Loop::new(input)
  input[0] = input[1]
  let backend = backend()
  ignore(driver.advance(backend))
  assert_eq(backend.video[0], 2.75)
  let invalid = [
    [],
    [input[1], input[1]],
    [
      {
        video_timestep: 1.0,
        audio_timestep: 1.0,
        video_sigma: 0.0,
        audio_sigma: 0.0,
        video_next_sigma: 0.0,
        audio_next_sigma: 0.0,
      },
    ],
  ]
  for schedule in invalid {
    try @joint_diffusion_loop.Loop::new(schedule) catch {
      @joint_diffusion_loop.LoopError::InvalidSchedule => ()
      _ => fail("wrong schedule error")
    } noraise {
      _ => fail("invalid schedule accepted")
    }
  }
}

///|
struct CancellingBackend {
  driver : @joint_diffusion_loop.Loop
  mut completed : Bool
}

///|
impl @joint_diffusion_loop.JointBackend for CancellingBackend with fn execute_step(
  self,
  _index,
  _step,
) {
  self.driver.cancel()
  // A callback cannot recursively submit another step, even after cancellation.
  try self.driver.advance(self) catch {
    @joint_diffusion_loop.LoopError::ReentrantProgress => ()
    _ => fail("wrong recursive progress error")
  } noraise {
    _ => fail("recursive dispatch accepted")
  }
  self.completed = true
}

///|
test "in-flight cancellation waits for synchronous work to complete" {
  let driver = @joint_diffusion_loop.Loop::new(steps())
  let backend = CancellingBackend::{ driver, completed: false, }
  assert_eq(driver.advance(backend), Cancelled)
  assert_true(backend.completed)
  assert_eq(driver.completed_steps(), 1)
}

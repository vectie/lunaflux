///|
struct CallbackFrame {
  mut callback : () -> Unit raise
  mut releases : Int
}

///|
impl @joint_diffusion_execution.PreparedFrame for CallbackFrame with fn execute(
  self,
) {
  (self.callback)()
}

///|
impl @joint_diffusion_execution.PreparedFrame for CallbackFrame with fn drain(
  _self,
) {
  ()
}

///|
impl @joint_diffusion_execution.PreparedFrame for CallbackFrame with fn release(
  self,
) {
  self.releases += 1
}

///|
test "close inside active execution cannot release any device frame" {
  let first : CallbackFrame = { callback: () => (), releases: 0, }
  let second : CallbackFrame = { callback: () => (), releases: 0, }
  let execution = @joint_diffusion_execution.Execution::new(plan(), [
    first, second,
  ])
  first.callback = () => {
    try execution.close() catch {
      @joint_diffusion_execution.ExecutionError::ReentrantClose => ()
      _ => fail("wrong reentrant close error")
    } noraise {
      _ => fail("released active execution")
    }
    assert_eq(first.releases, 0)
    assert_eq(second.releases, 0)
  }
  assert_eq(execution.advance(), Running)
  assert_eq(execution.advance(), Completed)
  execution.close()
  assert_eq(first.releases, 1)
  assert_eq(second.releases, 1)
}

///|
test "pipeline cancellation inside a frame is sticky and cannot release active work" {
  let frames : Array[CallbackFrame] = [
    for _ in 0..<5 => { callback: () => (), releases: 0, }
  ]
  let pipeline = @joint_diffusion_execution.Pipeline::new(
    plan(),
    frames[0:1],
    frames[1:3],
    frames[3],
    frames[4],
  )
  frames[0].callback = () => {
    pipeline.cancel()
    try pipeline.advance() catch {
      @joint_diffusion_execution.PipelineError::Reentrant => ()
      _ => fail("wrong reentrant advance error")
    } noraise {
      _ => fail("pipeline reentered")
    }
    try pipeline.close() catch {
      @joint_diffusion_execution.PipelineError::Reentrant => ()
      _ => fail("wrong reentrant close error")
    } noraise {
      _ => fail("active pipeline released")
    }
    for frame in frames {
      assert_eq(frame.releases, 0)
    }
  }
  assert_true(pipeline.advance() == Cancelled)
  assert_eq(pipeline.completed_frames(), 1)
  pipeline.close()
  pipeline.close()
  for frame in frames {
    assert_eq(frame.releases, 1)
  }
}

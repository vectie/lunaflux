///|
struct PipelineFrame {
  id : Int
  log : Array[Int]
  fail : Bool
  mut fail_drain : Bool
  mut fail_release : Bool
}

///|
impl @joint_diffusion_execution.PreparedFrame for PipelineFrame with fn execute(
  self,
) {
  self.log.push(self.id)
  if self.fail {
    raise Injected::Run
  }
}

///|
impl @joint_diffusion_execution.PreparedFrame for PipelineFrame with fn drain(
  self,
) {
  if self.fail_drain {
    self.fail_drain = false
    raise Injected::Release
  }
  self.log.push(100 + self.id)
}

///|
impl @joint_diffusion_execution.PreparedFrame for PipelineFrame with fn release(
  self,
) {
  if self.fail_release {
    self.fail_release = false
    raise Injected::Release
  }
  self.log.push(200 + self.id)
}

///|
fn pipeline_frames(fail? : Int = -1) -> Array[PipelineFrame] {
  let log = []
  [
    for id in 0..<5 => {
      { id, log, fail: id == fail, fail_drain: false, fail_release: false, }
    }
  ]
}

///|
test "failed drain retains current stage and closes no upstream dependency" {
  let frames = pipeline_frames()
  let p = @joint_diffusion_execution.Pipeline::new(
    plan(),
    frames[0:1],
    frames[1:3],
    frames[3],
    frames[4],
  )
  frames[4].fail_drain = true
  try p.close() catch {
    _ => ()
  } noraise {
    _ => fail("drain failure swallowed")
  }
  assert_eq(frames[0].log.length(), 0)
  try p.advance() catch {
    _ => ()
  } noraise {
    _ => fail("closing request advanced")
  }
  p.close()
  inspect(
    frames[0].log,
    content="[104, 204, 103, 203, 102, 202, 101, 201, 100, 200]",
  )
}

///|
test "full request orders conditioning denoising and both decoders" {
  let frames = pipeline_frames()
  let p = @joint_diffusion_execution.Pipeline::new(
    plan(),
    frames[0:1],
    frames[1:3],
    frames[3],
    frames[4],
  )
  assert_true(p.advance() == Denoising)
  assert_true(p.advance() == Denoising)
  assert_true(p.advance() == VideoDecode)
  assert_true(p.advance() == AudioDecode)
  assert_true(p.advance() == Completed)
  assert_eq(p.completed_frames(), 5)
  inspect(frames[0].log, content="[0, 1, 2, 3, 4]")
  try p.advance() catch {
    _ => ()
  } noraise {
    _ => fail("completed request replayed")
  }
  frames[4].fail_release = true
  try p.close() catch {
    _ => ()
  } noraise {
    _ => fail("release failure lost")
  }
  p.close()
  p.close()
  assert_true(p.stage() == Closed)
  inspect(
    frames[0].log,
    content="[0, 1, 2, 3, 4, 104, 204, 103, 203, 102, 202, 101, 201, 100, 200]",
  )
}

///|
test "pipeline partial failure cannot publish output or run later stages" {
  for broken in 0..<5 {
    let frames = pipeline_frames(fail=broken)
    let p = @joint_diffusion_execution.Pipeline::new(
      plan(),
      frames[0:1],
      frames[1:3],
      frames[3],
      frames[4],
    )
    for _ in 0..<broken {
      ignore(p.advance())
    }
    try p.advance() catch {
      _ => ()
    } noraise {
      _ => fail("frame failure swallowed")
    }
    assert_true(p.stage() == Failed)
    assert_eq(frames[0].log.length(), broken + 1)
    try p.advance() catch {
      _ => ()
    } noraise {
      _ => fail("failed request replayed")
    }
    p.close()
  }
}

///|
test "pipeline cancellation stops later modality and count admission is exact" {
  let frames = pipeline_frames()
  let p = @joint_diffusion_execution.Pipeline::new(
    plan(),
    frames[0:1],
    frames[1:3],
    frames[3],
    frames[4],
  )
  ignore(p.advance())
  p.cancel()
  assert_true(p.stage() == Cancelled)
  try p.advance() catch {
    _ => ()
  } noraise {
    _ => fail("cancelled request advanced")
  }
  p.close()
  try
    @joint_diffusion_execution.Pipeline::new(
      plan(),
      frames[0:0],
      frames[1:3],
      frames[3],
      frames[4],
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("missing conditioning accepted")
  }
  try
    @joint_diffusion_execution.Pipeline::new(
      plan(),
      frames[0:1],
      frames[1:2],
      frames[3],
      frames[4],
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("missing denoise step accepted")
  }
}

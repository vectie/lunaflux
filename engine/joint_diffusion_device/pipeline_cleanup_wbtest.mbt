///|
priv suberror PendingCleanupFailure {
  InjectedRelease
}

///|
priv struct PendingCleanupProbe {
  id : Int
  log : Array[Int]
  mut fail_release : Bool
}

///|
impl @joint_diffusion_execution.PreparedFrame for PendingCleanupProbe with fn execute(
  _self,
) {
  ()
}

///|
impl @joint_diffusion_execution.PreparedFrame for PendingCleanupProbe with fn drain(
  self,
) {
  self.log.push(self.id * 10 + 1)
}

///|
impl @joint_diffusion_execution.PreparedFrame for PendingCleanupProbe with fn release(
  self,
) {
  self.log.push(self.id * 10 + 2)
  if self.fail_release {
    self.fail_release = false
    raise InjectedRelease
  }
}

///|
test "partial pipeline preparation retries release without redraining a partially released owner" {
  let log = []
  let program = PipelineProgram::new()
  program.started = true
  for id in 1..<=3 {
    let probe : PendingCleanupProbe = { id, log, fail_release: id == 3, }
    program.pending.push(@joint_diffusion_execution.PreparedStage::own(probe))
  }
  try program.close() catch {
    InjectedRelease => ()
    _ => fail("wrong cleanup error")
  } noraise {
    _ => fail("injected release failure ignored")
  }
  assert_eq(log, [31, 32])
  assert_false(program.outputs_ready())
  program.close()
  assert_eq(log, [31, 32, 32, 21, 22, 11, 12])
  program.close()
  assert_eq(log, [31, 32, 32, 21, 22, 11, 12])
}

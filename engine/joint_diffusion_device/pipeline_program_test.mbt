///|
test "unprepared whole request cannot execute or publish outputs" {
  let request = @joint_diffusion_device.PipelineProgram::new()
  assert_false(request.outputs_ready())
  request.cancel()
  try request.advance() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared pipeline advanced")
  }
  request.close()
  request.close()
  assert_false(request.outputs_ready())
}

///|
test "unprepared vision request cannot submit poll or expose output" {
  let p = @vision.VisionPlan::new(
    [{ time: 1, height: 2, width: 2, }],
    max_workspace_bytes=1000000L,
    max_output_bytes=1000000L,
  )
  let request = VisionRequest::new(build_source(p, prefix="v"))
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared submit")
  }
  try request.poll() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared poll")
  }
  try request.with_output((_, _) => ()) catch {
    _ => ()
  } noraise {
    _ => fail("unprepared output")
  }
  request.cancel()
  assert_true(request.phase == Cancelled)
  request.close()
  request.close()
  assert_true(request.phase == Closed)
  let weights = VisionWeights::new()
  weights.close()
  weights.close()
}

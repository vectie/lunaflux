///|
test "heterogeneous stages retain execution order and concrete release owners" {
  let frames = pipeline_frames()
  let audio : CallbackFrame = {
    callback: () => frames[0].log.push(4),
    releases: 0,
  }
  let conditioning = [@joint_diffusion_execution.PreparedStage::own(frames[0])]
  let predictions = [
    @joint_diffusion_execution.PreparedStage::own(frames[1]),
    @joint_diffusion_execution.PreparedStage::own(frames[2]),
  ]
  let pipeline = @joint_diffusion_execution.Pipeline::new(
    plan(),
    conditioning,
    predictions,
    @joint_diffusion_execution.PreparedStage::own(frames[3]),
    @joint_diffusion_execution.PreparedStage::own(audio),
  )
  for _ in 0..<5 {
    ignore(pipeline.advance())
  }
  assert_true(pipeline.stage() == Completed)
  inspect(frames[0].log, content="[0, 1, 2, 3, 4]")
  pipeline.close()
  assert_eq(audio.releases, 1)
}

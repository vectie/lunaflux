///|
test "MTP and DSpark auxiliary stages remain explicitly ordered" {
  let mtp = try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
    2,
  )
  assert_eq(auxiliary_stage_order(mtp, limits=test_limits()), [
    MtpInputProjection,
    AuxiliaryDecoderStage(0),
    AuxiliaryDecoderStage(1),
    AuxiliaryOutputHead,
  ])
  let dspark = try! @advanced_decoder_execution_plan.DsparkExecutionPlan::new(
    block_size=5,
    noise_token_id=99,
    target_layer_ids=[2, 3, 4],
    markov_rank=8,
    decoder_layer_count=5,
  )
  let plan = try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction_with_dspark(
    layer_count=3,
    dspark~,
  )
  assert_eq(auxiliary_stage_order(plan, limits=test_limits()), [
    DsparkMainProjection,
    DsparkNoiseEmbedding,
    AuxiliaryDecoderStage(0),
    AuxiliaryDecoderStage(1),
    AuxiliaryDecoderStage(2),
    AuxiliaryOutputHead,
    DsparkMarkovAdjustment,
    DsparkConfidence,
  ])
  let low_stage_limit = try! ReferenceLimits::new(
    max_values=32,
    max_width=8,
    max_iterations=2,
  )
  try auxiliary_stage_order(plan, limits=low_stage_limit) catch {
    InvalidAuxiliaryOrder => ()
    _ => fail("unexpected auxiliary-order bound error")
  } noraise {
    _ => fail("DSpark order exceeded its explicit stage bound")
  }
}

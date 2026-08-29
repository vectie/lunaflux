///|
async fn qwen3_plan_check(
  args : ArrayView[String],
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(args.length() == 6, "plan-check arguments are invalid")
  let metadata = qwen3_load_metadata(args[2], args[3], args[4])
  qwen3_require(
    metadata.spec().layer_count() == 28,
    "Qwen3 serving plan does not contain 28 layers",
  )
  let plan = @qwen3.build_paged(
    metadata,
    batch=@qwen3.Qwen3PagedBatchEnvelope::single_row(),
  ) catch {
    _ => raise CandidateMismatch("Qwen3 serving plan construction failed")
  }
  qwen3_require(
    plan.identity().plan().as_hex() == args[5],
    "Qwen3 serving plan digest differs",
  )
  println(
    "outcome=qwen3-serving-plan-pass model_family=qwen3 layer_count=28 model_plan_sha256=" +
    args[5] +
    " batch_capacity=1",
  )
}

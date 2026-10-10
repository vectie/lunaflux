///|
/// Startup file adapter for the pure, model-neutral serial prompt plan.
/// Read each bounded frame once. Returned immutable bytes outlive the caller's
/// root; this adapter retains no file descriptor, CUDA state or payload hash.
pub fn load_generation_prompt(
  root : @approved_fs.ApprovedRoot,
  count : Int,
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  maximum_new_tokens~ : Int,
  maximum_context~ : Int,
  maximum_bytes~ : Int64,
  model_generation~ : UInt64,
) -> @prompt.PreparedSerialPrompt raise {
  guard maximum_new_tokens > 0 &&
    maximum_new_tokens < maximum_context &&
    count > 0 &&
    count <= maximum_context - maximum_new_tokens &&
    maximum_bytes > 0L else {
    fail("invalid prompt replay envelope")
  }
  let capacity = @wire.PlanFrameBuffer::new(limits, inference).capacity()
  let frames : Array[Bytes] = []
  // Invariant: remaining >= 0, and retained frame bytes equal budget-remaining.
  // Bounded reads reject a larger file before its payload allocation.
  for index in 0..<count; remaining = maximum_bytes {
    let bytes = {
      let file = root.open_file(
        @approved_fs.ApprovedRelativeLocator::new("plan-\{index}.bin"),
      )
      defer file.close()
      file.read_immutable_snapshot(
        maximum_bytes=remaining.min(capacity.to_int64()),
      )
    }
    frames.push(bytes)
    continue remaining - bytes.length().to_int64()
  }
  @prompt.PreparedSerialPrompt::new(
    frames,
    limits,
    inference,
    maximum_new_tokens~,
    maximum_context~,
    maximum_encoded_bytes=maximum_bytes,
    model_generation~,
  )
}

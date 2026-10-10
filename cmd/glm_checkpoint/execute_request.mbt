///|
// Terminal-only marker, emitted only after deterministic context close accepts
// that every child has retired. This is not request success telemetry.
fn close_rank_context(context : @device.Context, role : String) -> Unit raise {
  context.close()
  println("rank_cleanup: role=\{role} context_closed=true")
}

///|
// Request-local continuation. Rank, weights, transport and CUDA ownership stay
// with the enclosing session; request retirement only releases its frontiers.
async fn execute_ingress_request(
  startup : Startup,
  frame_root : @approved_fs.ApprovedRoot,
  prepared : @prompt.PreparedSerialPrompt?,
  count : Int,
  pipeline : @pipeline.TwoRankDecoderPipeline,
  clock : @clock.MonotonicClock,
  generate~ : Bool,
  maximum_new_tokens~ : Int,
  stop~ : Int,
  output_directory~ : String,
  first_epoch~ : UInt64,
) -> UInt64 {
  let limits = worker_limits(startup)
  let inference = inference_limits(startup)
  let plans = @wire.PlanFrameBuffer::new(limits, inference)
  let completions = @wire.CompletionFrameBuffer::new(limits, inference)
  let scratch = FixedArray::make(plans.capacity(), b'\x00')
  let output = FixedArray::make(completions.capacity(), b'\x00')
  let generation : Ref[@generation.SerialGreedyGeneration?] = Ref(None)
  let started = clock.now_millis()
  let mut next_epoch = first_epoch
  for index in 0..<count {
    let input = match prepared {
      Some(prompt) => prompt.frame_bytes(index)
      None =>
        read_bounded(
          frame_root,
          "plan-\{index}.bin",
          plans.capacity().to_int64(),
        )
    }
    scratch.fill(b'\x00')
    for i, byte in input {
      scratch[i] = byte
    }
    let plan = plans.load(scratch, input.length())
    pipeline.begin(
      plan,
      completions.writer_for(plan),
      epoch=next_epoch,
      local_now_millis=clock.now_millis(),
    )
    while !pipeline.progress(local_now_millis=clock.now_millis()) {
      @async.pause()
    }
    next_epoch += 1UL
    let completion = pipeline.completion()
    completion.copy_to(output, destination_offset=0)
    @fs.write_file(
      output_directory + "/completion-\{index}.bin",
      Bytes::from_array(output[:completion.length()].to_owned()),
      create_mode=CreateNew,
    )
    let entry = completion.entry(0)
    guard entry.kind() != @protocol.CompletionEntryKind::FailedCompletion else {
      fail("model execution returned a failed completion")
    }
    if generate && index == count - 1 {
      generation.val = Some(
        @generation.SerialGreedyGeneration::new(
          plan,
          completion,
          limits,
          inference,
          maximum_new_tokens~,
          maximum_context=startup.history,
          stop_tokens=if stop == -1 { [] } else { [stop] },
        ),
      )
    }
    if entry.kind() == @protocol.CompletionEntryKind::PrefillCompletion {
      println("frame=\{index} processed=\{entry.processed_tokens()}")
    } else {
      println(
        "frame=\{index} token=\{entry.token_id()} processed=\{entry.processed_tokens()}",
      )
    }
  }
  if generate {
    guard generation.val is Some(owner) else { fail("no completed prompt") }
    while owner.finish_reason() is Running {
      let plan = owner.next_decode()
      pipeline.begin(
        plan,
        completions.writer_for(plan),
        epoch=next_epoch,
        local_now_millis=clock.now_millis(),
      )
      while !pipeline.progress(local_now_millis=clock.now_millis()) {
        @async.pause()
      }
      next_epoch += 1UL
      owner.accept(pipeline.completion())
    }
  }
  pipeline.release_request(local_now_millis=clock.now_millis())
  while !pipeline.progress(local_now_millis=clock.now_millis()) {
    @async.pause()
  }
  if generation.val is Some(owner) {
    // Diagnostic output is rendered only after both ranks retire the request.
    let rendered = StringBuilder()
    for index in 0..<owner.token_count() {
      rendered <+ "\{owner.token_at(index)}\n"
    }
    @fs.write_file(
      output_directory + "/tokens.txt",
      rendered.to_string(),
      create_mode=CreateNew,
    )
    println(
      "generated=\{owner.token_count()} finish=\{Repr(owner.finish_reason())}",
    )
  }
  println("generation_millis=\{clock.now_millis() - started}")
  next_epoch
}

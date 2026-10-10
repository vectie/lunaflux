///|
fn prepare_prompt_sequence(
  root : @approved_fs.ApprovedRoot,
  requests : Int,
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  maximum_new_tokens~ : Int,
  maximum_context~ : Int,
  maximum_bytes~ : Int64,
) -> ReadOnlyArray[@prompt.PreparedSerialPrompt] raise {
  @prompt_file.load_generation_sequence_file(
    root,
    requests,
    limits,
    inference,
    maximum_new_tokens~,
    maximum_context~,
    maximum_bytes~,
    model_generation=1UL,
  )
}

///|
// Request-local continuation state. Resident execution owners belong to the
// outer rank scope; ordinary request retirement does not close their queues,
// activation channels, weights, AOT module or context.
async fn execute_ingress_request(
  startup : Startup,
  frame_root : @approved_fs.ApprovedRoot,
  prepared : @prompt.PreparedSerialPrompt?,
  count : Int,
  pipeline : @pipeline.TwoRankDecoderPipeline,
  client : @control.DecoderStageClient,
  verifier : @verification.TwoRankGreedyVerification?,
  clock : @clock.MonotonicClock,
  generate~ : Bool,
  speculative_enabled~ : Bool,
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
  let speculative : Ref[@generation.SpeculativeGreedyGeneration?] = Ref(None)
  let generation_start = clock.now_millis()
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
      if speculative_enabled {
        speculative.val = Some(
          @generation.SpeculativeGreedyGeneration::new(
            plan,
            completion,
            limits,
            inference,
            maximum_new_tokens~,
            maximum_context=startup.history,
            stop_tokens=if stop == -1 { [] } else { [stop] },
          ),
        )
      } else {
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
    }
    if entry.kind() == @protocol.CompletionEntryKind::PrefillCompletion {
      println("frame=\{index} processed=\{entry.processed_tokens()}")
    } else {
      println(
        "frame=\{index} token=\{entry.token_id()} processed=\{entry.processed_tokens()}",
      )
    }
  }
  let mut verification_released = false
  @stdio.stdout.write(
    "generation_started: speculative=\{speculative_enabled}\n",
  )
  if speculative.val is Some(owner) && owner.finish_reason() is Running {
    guard verifier is Some(session) else {
      fail("missing resident verification session")
    }
    next_epoch = generate_speculative(
      startup,
      owner,
      client,
      session,
      clock,
      first_epoch=next_epoch,
    )
    pipeline.retire_publication()
    verification_released = true
  } else if generate && !speculative_enabled {
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
  if !verification_released {
    pipeline.release_request(local_now_millis=clock.now_millis())
    while !pipeline.progress(local_now_millis=clock.now_millis()) {
      @async.pause()
    }
  }
  if generation.val is Some(owner) {
    write_generated(
      output_directory,
      owner.token_count(),
      owner.finish_reason(),
      index => owner.token_at(index),
    )
  }
  if speculative.val is Some(owner) {
    write_generated(
      output_directory,
      owner.token_count(),
      owner.finish_reason(),
      index => owner.token_at(index),
    )
  }
  println("generation_millis=\{clock.now_millis() - generation_start}")
  next_epoch
}

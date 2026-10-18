///|
fn batch_control_limits() -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=4,
    max_capabilities=4,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
// Each request has one committed frontier even when its ordinal changes.
fn batch_control_plan(
  entries : ArrayView[(UInt64, UInt64, Int, Bool)],
) -> (
  @protocol.SubmittedSchedulePlan,
  @wire.ValidatedPlanFrame,
  @wire.CompletionFrameWriter,
) raise {
  let limits = batch_control_limits()
  let inference = serial_inference_limits()
  let owner = @protocol.SchedulePlanBuffer::new(limits)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=4,
      max_run_pages=4,
      max_active_references_per_page=4,
      max_cached_references_per_page=4,
    ),
  )
  for index, (id, generation, position, final_prefill) in entries {
    let request_id = @inference.RequestId::new(id)
    let request_generation = @inference.RequestGeneration::from_wire(generation)
    let pages = [allocator.allocate_one()]
    let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
    let completion_slot = @protocol.CompletionSlot::new(index, limits)
    if position > 0 {
      owner.append_decode(
        request_id~,
        request_generation~,
        sequence_token_position=position,
        input_token=7,
        pages~,
        capabilities~,
        sampling=@inference.SamplingParameters::greedy(),
        sampling_seed=@inference.SamplingSeed::new(id),
        sample_index=position,
        completion_slot~,
      )
    } else if final_prefill {
      owner.append_final_prefill(
        request_id~,
        request_generation~,
        sequence_token_start=0,
        tokens=[2, 3],
        pages~,
        capabilities~,
        sampling=@inference.SamplingParameters::greedy(),
        sampling_seed=@inference.SamplingSeed::new(id),
        sample_index=1,
        completion_slot~,
      )
    } else {
      owner.append_prefill(
        request_id~,
        request_generation~,
        sequence_token_start=0,
        tokens=[2, 3],
        pages~,
        capabilities~,
        completion_slot~,
      )
    }
  }
  let total = entries.fold(init=0, (sum, entry) => {
    sum + (if entry.2 > 0 { 1 } else { 2 })
  })
  let submitted = owner.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=9UL,
    token_budget=total,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=9UL,
  )
  let frame = @wire.PlanFrameBuffer::new(limits, inference).encode_plan(
    submitted,
  )
  let writer = @wire.CompletionFrameBuffer::new(limits, inference).writer_for(
    frame,
  )
  (submitted, frame, writer)
}

///|
async test "batched transport publishes selected rows and retires only identified state" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let rank = batch_rank_fixture()
  defer (rank.close() catch { _ => () })
  rank.prepare_ports(context, maximum_device_bytes=rank.required_device_bytes())
  context.copy_from_fixed_host(
    rank.sampled_tokens(),
    [7, 0, 0, 0, 8, 0, 0, 0],
    0L,
    0L,
    8L,
  )
  rank.bind_executor(context, submit=_ => (), poll=() => true, close=() => (), append_descriptors=[])
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let limits = batch_control_limits()
  let budget = @stage_control.required_buffer_bytes(limits)
  let client = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_output_rows=2,
    batched_outputs=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
  )
  defer (client.close() catch { _ => () })
  let server = @stage_control.DecoderStageServer::new_batched(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
  )
  defer (server.close() catch { _ => () })
  // A partial prefill has no sampled row, even though it contains two tokens.
  let (_, first, _) = batch_control_plan([
    (7UL, 0UL, 0, false),
    (8UL, 0UL, 0, true),
  ])
  client.prepare(first, epoch=1UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(client.output_rows(), 1)
  assert_eq(client.sampled_token_at(0), 7)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  let (_, reordered, _) = batch_control_plan([
    (8UL, 0UL, 2, true),
    (7UL, 0UL, 2, true),
  ])
  client.prepare(reordered, epoch=2UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(client.output_rows(), 2)
  assert_eq(client.sampled_token_at(0), 7)
  assert_eq(client.sampled_token_at(1), 8)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  // Release request 7, not ordinal zero (request 8 in the preceding frame).
  client.release_one(7UL, 0UL, local_now_millis=0UL)
  control_exchange(client, server)
  let (_, reused, _) = batch_control_plan([
    (7UL, 1UL, 0, true),
    (8UL, 0UL, 3, true),
  ])
  client.prepare(reused, epoch=3UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  client.release_one(7UL, 0UL, local_now_millis=0UL)
  try control_exchange(client, server) catch {
    @batch_frame.BatchFrameError::GenerationMismatch => ()
    error => raise error
  } noraise {
    _ => fail("stale release retired a reused request")
  }
  server.close()
  client.close()
  rank.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

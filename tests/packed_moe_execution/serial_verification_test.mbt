///|
fn verification_host() -> @serial_frame.SerialDecoderFrame raise {
  @serial_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=16,
    vocabulary_size=16,
    model_generation=9UL,
    with_positions=true,
    all_position_outputs=true,
  )
}

///|
fn store_verification_samples(
  context : @device.Context,
  ports : @serial_worker.SerialDecoderWorker,
  samples : ArrayView[Int],
) -> Unit raise {
  let bytes = FixedArray::make(ports.output_capacity() * 4, b'\x00')
  for row, sample in samples {
    let bits = sample.reinterpret_as_uint()
    for i in 0..<4 {
      bytes[row * 4 + i] = ((bits >> (i * 8)) & 255U).to_byte()
    }
  }
  context.copy_from_fixed_host(
    ports.sampled_tokens(),
    bytes,
    0L,
    0L,
    bytes.length().to_int64(),
  )
}

///|
test "greedy prefix decisions have no warmed heap allocation" {
  let inputs = [10, 11, 12, 13]
  let outputs = [11, 12, 7, 15]
  let stops : Array[Int] = []
  begin_allocations()
  let accepted = for _ in 0..<256; total = 0 {
    let prefix = @greedy_verification.verify_greedy_prefix(
      inputs,
      outputs,
      remaining_new_tokens=10,
      vocabulary_size=16,
      stop_tokens=stops,
    )
    continue total + prefix.accepted_input_rows()
  } nobreak {
    total
  }
  let allocations = end_allocations()
  assert_eq(accepted, 768)
  assert_eq(allocations, 0UL)
}

///|
test "rank verification returns every sample and rolls back host frontier without allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let host = verification_host()
  let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=true)
  let ports = rank.ports()
  assert_eq(ports.output_capacity(), 4)
  assert_eq(ports.required_device_bytes(), 108L)
  ports.prepare_ports(context, maximum_device_bytes=108L)
  assert_eq(ports.sampled_tokens().byte_count(), 16L)
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    ports.input(Counts),
  )
  bind_rank_queue(rank, context, queue, [])
  defer {
    rank.close() catch {
      _ => ()
    }
    function.close() catch {
      _ => ()
    }
    aot.close() catch {
      _ => ()
    }
    stream.close() catch {
      _ => ()
    }
    context.close() catch {
      _ => ()
    }
  }
  let (_, prompt, prompt_writer) = serial_wire_plan(7UL, produces=false)
  rank.begin(prompt, prompt_writer)
  while !rank.inputs_ready() {
    ()
  }
  rank.submit()
  while !rank.progress() {
    ()
  }
  assert_eq(rank.output_rows(), 0)
  rank.commit(pipeline_failed=false)
  assert_eq(host.processed_tokens(), 2)
  store_verification_samples(context, ports, [7, 9])
  let (_, verify, verify_writer) = serial_wire_plan(7UL, position=2, tokens=[
    4, 5,
  ])
  let waits = blocking_syncs()
  begin_allocations()
  rank.begin(verify, verify_writer)
  while !rank.inputs_ready() {
    ()
  }
  let uploaded_allocations = end_allocations()
  begin_allocations()
  rank.submit()
  while !rank.progress() {
    ()
  }
  let execution_allocations = end_allocations()
  begin_allocations()
  assert_false(rank.execution_failed())
  assert_eq(rank.output_rows(), 2)
  assert_eq(rank.sampled_token_at(0), 7)
  assert_eq(rank.sampled_token_at(1), 9)
  let sample_allocations = end_allocations()
  // Device restoration belongs to the snapshot owner; this fixture measures
  // only retired sample transport and the corresponding host transaction.
  begin_allocations()
  rank.rollback_retired()
  let rollback_allocations = end_allocations()
  assert_eq(host.processed_tokens(), 2)
  assert_eq(uploaded_allocations, 0UL)
  assert_eq(execution_allocations, 0UL)
  assert_eq(sample_allocations, 0UL)
  assert_eq(rollback_allocations, 0UL)
  assert_eq(blocking_syncs(), waits)
  let (_, replay, replay_writer) = serial_wire_plan(
    7UL,
    position=2,
    tokens=[4],
    produces=false,
  )
  rank.begin(replay, replay_writer)
  while !rank.inputs_ready() {
    ()
  }
  rank.submit()
  while !rank.progress() {
    ()
  }
  rank.commit(pipeline_failed=false)
  assert_eq(host.processed_tokens(), 3)
  rank.close()
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "invalid nonfirst verification sample is an execution failure, not a rollback" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let host = verification_host()
  let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=true)
  let ports = rank.ports()
  ports.prepare_ports(context, maximum_device_bytes=108L)
  store_verification_samples(context, ports, [7, 16])
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    ports.input(Counts),
  )
  bind_rank_queue(rank, context, queue, [])
  let (_, plan, writer) = serial_wire_plan(7UL)
  rank.begin(plan, writer)
  while !rank.inputs_ready() {
    ()
  }
  rank.submit()
  while !rank.progress() {
    ()
  }
  assert_true(rank.execution_failed())
  try rank.rollback_retired() catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("invalid later sample treated as an accepted transaction")
  }
  try rank.sampled_token_at(0) catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("failure exposed a partial successful sample")
  }
  rank.commit(pipeline_failed=false)
  assert_true(host.is_faulted())
  rank.close()
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

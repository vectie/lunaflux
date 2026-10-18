///|
test "sampling binds only after executor ownership and closes exactly once" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let rank = batch_rank_fixture()
  defer (rank.close() catch { _ => () })
  rank.prepare_ports(context, maximum_device_bytes=rank.required_device_bytes())
  let closes = Ref::new(0)
  let sampler = @batch_worker.BatchedSampling::new(
    maximum_outputs=2,
    vocabulary_size=16,
    prepare=_ => (),
    begin=() => (),
    poll=() => true,
    token_at=_ => 0,
    close=() => closes.val += 1,
  )
  try rank.bind_sampling(sampler) catch {
    @batch_frame.BatchFrameError::InvalidCapacity => ()
    error => raise error
  } noraise {
    _ => fail("sampling bound before execution owner")
  }
  assert_eq(closes.val, 0)
  rank.bind_executor(context, submit=_ => (), poll=() => true, close=() => (), append_descriptors=[])
  rank.bind_sampling(sampler)
  rank.close()
  rank.close()
  assert_eq(closes.val, 1)
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "batched full logits preserve canonical seeded sampling without step allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let rank = batch_rank_fixture()
  defer (rank.close() catch { _ => () })
  rank.prepare_ports(context, maximum_device_bytes=rank.required_device_bytes())
  rank.bind_executor(context, submit=_ => (), poll=() => true, close=() => (), append_descriptors=[])
  let logits = context.allocate(128L)
  defer (logits.close() catch { _ => () })
  let bytes = FixedArray::make(128, b'\x00')
  let values = Array::makei(16, i => (i % 5).to_double() / 8.0)
  for i in 0..<32 {
    let bits = Float::from_double(values[i % 16]).reinterpret_as_uint()
    for byte in 0..<4 {
      bytes[i * 4 + byte] = ((bits >> (byte * 8)) & 255U).to_byte()
    }
  }
  context.copy_from_fixed_host(logits, bytes, 0L, 0L, 128L)
  let sampler = @batch_sampling.BatchedLogitSampler::new(
    context,
    logits,
    vocabulary_size=16,
    maximum_outputs=2,
    maximum_host_bytes=4096L,
  )
  defer (sampler.close() catch { _ => () })
  let (_, first, first_writer) = batch_control_plan(
    [(7UL, 0UL, 0, true), (8UL, 0UL, 0, true)],
    stochastic=true,
  )
  try rank.begin(first, first_writer) catch {
    @batch_frame.BatchFrameError::InvalidCapacity => ()
    error => raise error
  } noraise {
    _ => fail("stochastic plan silently used a greedy-only binding")
  }
  rank.bind_sampling(sampler.binding())
  let limits = @sampling.SamplingLimits::new(2, 16)
  let scratch = @sampling.SamplerScratch::new(16, limits)
  let parameters = @inference.SamplingParameters::sample(
    temperature=0.75,
    top_k=3,
    top_p=0.9,
    serial_inference_limits(),
  )
  let expected = [7UL, 8UL].map(id => {
    @sampling.stochastic_sample_at(
      values[:],
      parameters,
      limits,
      @inference.SamplingSeed::new(id),
      1,
      scratch,
    ).token_id()
  })
  let plans = Array::makei(16, _ => {
    batch_control_plan(
      [(7UL, 0UL, 0, true), (8UL, 0UL, 0, true)],
      stochastic=true,
    )
  })
  let waits = blocking_syncs()
  begin_allocations()
  for (_, plan, writer) in plans {
    rank.begin(plan, writer)
    while !rank.inputs_ready() {

    }
    rank.submit()
    while !rank.progress() {

    }
    assert_false(rank.execution_failed())
    assert_eq(rank.samples()[0], expected[0])
    assert_eq(rank.samples()[1], expected[1])
    rank.commit(pipeline_failed=false)
    rank.release_request(7UL, 0UL)
    rank.release_request(8UL, 0UL)
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(blocking_syncs(), waits)
  rank.close()
  logits.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

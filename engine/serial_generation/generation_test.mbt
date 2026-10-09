///|
test "real completion tokens feed successive canonical decodes without altering the reserved recipe" {
  let initial = seed(position=3)
  let owner = generation(initial, 7)
  let first = owner.next_decode()
  assert_eq(first.token_at(0), 7)
  assert_eq(first.sequence_value(), 2UL)
  assert_eq(first.model_generation_value(), 9UL)
  let row = first.decode_row(0)
  assert_eq(row.request_id_value(), 7UL)
  assert_eq(row.request_generation_value(), 3UL)
  assert_eq(row.sequence_token_position(), 5)
  assert_eq(row.sample_index(), 5)
  assert_eq(row.sampling_seed_value(), 17UL)
  assert_eq(row.completion_slot(), 1)
  for index in 0..<initial.page_count() {
    assert_eq(first.page_index_at(index), initial.page_index_at(index))
    assert_eq(
      first.page_generation_at(index),
      initial.page_generation_at(index),
    )
  }
  for index in 0..<initial.capability_count() {
    assert_eq(
      first.capability_value_at(index),
      initial.capability_value_at(index),
    )
  }
  try owner.next_decode() catch {
    @serial_generation.SerialGenerationError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("inflight plan replaced")
  }
  owner.accept(complete(first, 11))
  let second = owner.next_decode()
  assert_eq(second.token_at(0), 11)
  assert_eq(second.decode_row(0).sequence_token_position(), 6)
  assert_eq(second.sequence_value(), 3UL)
  owner.accept(complete(second, 13))
  assert_eq(owner.finish_reason(), @serial_generation.GenerationFinish::Length)
  assert_eq(owner.token_count(), 3)
  for index, expected in [7, 11, 13] {
    assert_eq(owner.token_at(index), expected)
  }
  try owner.next_decode() catch {
    @serial_generation.SerialGenerationError::Finished => ()
    error => raise error
  } noraise {
    _ => fail("length limit ignored")
  }
}

///|
test "EOS on final prefill or decode stops without an extra submission" {
  let plan = seed()
  let immediate = generation(plan, 7, stops=[7])
  assert_eq(
    immediate.finish_reason(),
    @serial_generation.GenerationFinish::StopToken,
  )
  assert_eq(immediate.token_count(), 1)
  let owner = generation(plan, 7, stops=[11])
  let next = owner.next_decode()
  owner.accept(complete(next, 11))
  assert_eq(
    owner.finish_reason(),
    @serial_generation.GenerationFinish::StopToken,
  )
  assert_eq(owner.token_count(), 2)
  let single = generation(plan, 7, maximum=1)
  assert_eq(single.finish_reason(), @serial_generation.GenerationFinish::Length)
}

///|
test "context and sample limits are checked before allocating continuation or submitting a step" {
  let plan = seed(position=12)
  ignore(generation(plan, 7, maximum=2, context=16))
  for maximum in [0, 3, 4, 17] {
    try generation(plan, 7, maximum~, context=16) catch {
      @serial_generation.SerialGenerationError::InvalidLimit => ()
      error => raise error
    } noraise {
      _ => fail("out-of-bound generation admitted")
    }
  }
  for stops in [[7, 7], [-1], [32], [1, 2, 3]] {
    try generation(plan, 7, maximum=1, stops~) catch {
      @serial_generation.SerialGenerationError::InvalidLimit => ()
      error => raise error
    } noraise {
      _ => fail("invalid stop set admitted")
    }
  }
  try generation(seed(sequence=0xffffffffffffffffUL), 7) catch {
    @serial_generation.SerialGenerationError::InvalidLimit => ()
    error => raise error
  } noraise {
    _ => fail("sequence wrap admitted")
  }
}

///|
test "nonfinal prefill and stochastic seeds cannot silently enter a greedy-only continuation" {
  for plan in [seed(produces=false), seed(sample=true)] {
    try generation(plan, 7) catch {
      @wire.WorkerWireError::InvalidFrame(_) => ()
      error => raise error
    } noraise {
      _ => fail("unsupported seed admitted")
    }
  }
}

///|
test "foreign or failed completion poisons generation and cannot be retried as successful" {
  let initial = seed()
  try
    @serial_generation.SerialGreedyGeneration::new(
      initial,
      complete(seed(id=8UL), 11),
      limits(),
      inference(),
      maximum_new_tokens=3,
      maximum_context=16,
      stop_tokens=[],
    )
  catch {
    @wire.WorkerWireError::InvalidFrame(_) => ()
    error => raise error
  } noraise {
    _ => fail("foreign request completion seeded generation")
  }
  for failed in [true, false] {
    let owner = generation(initial, 7)
    let next = owner.next_decode()
    let bad = if failed {
      complete(next, 0, failed=true)
    } else {
      complete(seed(id=8UL), 11)
    }
    try owner.accept(bad) catch {
      @wire.WorkerWireError::InvalidFrame(_) => ()
      error => raise error
    } noraise {
      _ => fail("invalid completion accepted")
    }
    assert_eq(
      owner.finish_reason(),
      @serial_generation.GenerationFinish::Failed,
    )
    assert_eq(owner.token_count(), 1)
    try owner.accept(complete(next, 11)) catch {
      @serial_generation.SerialGenerationError::Busy => ()
      error => raise error
    } noraise {
      _ => fail("faulted continuation recovered without KV reset")
    }
  }
}

///|
test "cancelled inflight generation drains its exact completion without publishing another token" {
  let owner = generation(seed(), 7)
  let pending = owner.next_decode()
  owner.cancel()
  owner.cancel()
  owner.accept(complete(pending, 11))
  assert_eq(
    owner.finish_reason(),
    @serial_generation.GenerationFinish::Cancelled,
  )
  assert_eq(owner.token_count(), 1)
  try owner.next_decode() catch {
    @serial_generation.SerialGenerationError::Finished => ()
    error => raise error
  } noraise {
    _ => fail("cancelled continuation submitted")
  }
}

///|
test "wire continuation handles transport offsets and round trips through the ordinary decoder" {
  let initial = seed()
  let ordinary = @wire.PlanFrameBuffer::new(limits(), inference())
  let transport = FixedArray::make(ordinary.capacity() + 7, b'\xAA')
  let owner = @wire.PlanFrameBuffer::new_transport_owned(
    limits(),
    inference(),
    transport,
    frame_offset=7,
  )
  let next = owner.encode_greedy_decode_after(initial, complete(initial, 11))
  let scratch = FixedArray::make(ordinary.capacity(), b'\x00')
  next.copy_to(scratch, destination_offset=0)
  let decoded = ordinary.load(scratch, next.length())
  assert_eq(decoded.token_at(0), 11)
  assert_eq(decoded.decode_row(0).sequence_token_position(), 2)
  assert_eq(decoded.page_generation_at(1), initial.page_generation_at(1))
  for index in 0..<7 {
    assert_eq(transport[index], b'\xAA')
  }
  try
    ordinary.encode_greedy_decode_after(decoded, complete(decoded, 13))
  catch {
    @wire.WorkerWireError::InvalidFrame(_) => ()
    error => raise error
  } noraise {
    _ => fail("same-owner continuation overwrote its source frame")
  }
  assert_eq(decoded.token_at(0), 11)
}

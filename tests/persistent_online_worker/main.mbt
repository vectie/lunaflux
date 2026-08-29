///|
fn plan_request_id(frame : @worker_wire.ValidatedPlanFrame) -> UInt64 raise {
  if frame.prefill_count() + frame.decode_count() != 1 {
    raise Failure("persistent online fixture expected exactly one plan row")
  }
  if frame.prefill_count() == 1 {
    frame.prefill_row(0).request_id_value()
  } else {
    frame.decode_row(0).request_id_value()
  }
}

///|
fn validate_request_phase(
  request_id : UInt64,
  saw_first : Bool,
  saw_second : Bool,
  saw_third : Bool,
) -> (Bool, Bool, Bool) raise {
  if request_id == 950UL {
    if saw_second || saw_third {
      raise Failure("persistent online fixture returned to its first request")
    }
    (true, false, false)
  } else if request_id == 951UL {
    if !saw_first || saw_third {
      raise Failure("persistent online fixture skipped its first request")
    }
    (true, true, false)
  } else if request_id == 954UL {
    if !saw_second {
      raise Failure("persistent online fixture skipped its second request")
    }
    (true, true, true)
  } else {
    raise Failure("persistent online fixture received a foreign request")
  }
}

///|
fn run_worker() -> Unit raise {
  let channel = @process.open_inherited(
    @process.ProcessLimits::new(
      max_frame_bytes=67108864,
      io_timeout_millis=5000,
      shutdown_timeout_millis=5000,
    ),
  )
  let startup_owner = @worker_wire.StartupFrameBuffer::new()
  let startup_input = FixedArray::make(startup_owner.capacity(), b'\x00')
  let startup_length = channel.read_frame(startup_input)
  let configured = startup_owner.load(startup_input, startup_length)
  if configured.kind() != Configure {
    raise Failure("persistent online fixture startup kind mismatch")
  }
  let startup = configured.contract()
  let source_input = FixedArray::make(
    @worker_wire.bootstrap_source_capacity(),
    b'\x00',
  )
  let source_length = channel.read_frame(source_input)
  let source = @worker_wire.decode_bootstrap_source(source_input, source_length)
  if source.digest() != startup.bootstrap_source_digest() {
    raise Failure("persistent online fixture bootstrap source mismatch")
  }
  let limits = startup.worker_limits()
  let inference = startup.inference_limits()
  let plan_owner = @worker_wire.PlanFrameBuffer::new(limits, inference)
  let completion_owner = @worker_wire.CompletionFrameBuffer::new(
    limits, inference,
  )
  let startup_output = FixedArray::make(startup_owner.capacity(), b'\x00')
  let ready = startup_owner.encode(Ready, startup)
  let ready_length = ready.length()
  ready.copy_to(startup_output, destination_offset=0)
  channel.write_frame(startup_output, offset=0, byte_count=ready_length)
  let input = FixedArray::make(plan_owner.capacity(), b'\x00')
  let output = FixedArray::make(completion_owner.capacity(), b'\x00')
  let mut running = true
  let mut saw_first = false
  let mut saw_second = false
  let mut saw_third = false
  let mut frame_count = 0
  let mut last_sequence = 0UL
  while running {
    let length = channel.read_frame_or_eof(input) catch {
      _ => raise Failure("persistent online fixture frame receive failed")
    }
    if length == 0 {
      running = false
    }
    if running {
      let frame = plan_owner.load(input, length)
      if frame.sequence_value() != last_sequence + 1UL {
        raise Failure("persistent online fixture plan history reset")
      }
      last_sequence = frame.sequence_value()
      frame_count += 1
      let (next_first, next_second, next_third) = validate_request_phase(
        plan_request_id(frame),
        saw_first,
        saw_second,
        saw_third,
      )
      saw_first = next_first
      saw_second = next_second
      saw_third = next_third
      let writer = completion_owner.writer_for(frame)
      for index in 0..<frame.prefill_count() {
        let row = frame.prefill_row(index)
        if row.produces_token() {
          writer.append_final_prefill_completed(row, 0)
        } else {
          writer.append_prefill_completed(row)
        }
      }
      for index in 0..<frame.decode_count() {
        writer.append_decode_completed(frame.decode_row(index), 0)
      }
      let completion = writer.submit()
      let output_length = completion.length()
      completion.copy_to(output, destination_offset=0)
      channel.write_frame(output, offset=0, byte_count=output_length)
    }
  }
  if !saw_first || !saw_second || !saw_third || frame_count < 3 {
    raise Failure("persistent online fixture drained before all requests")
  }
}

///|
fn main {
  try! run_worker()
}

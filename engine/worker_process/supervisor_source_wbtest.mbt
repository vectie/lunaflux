///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "startup handshake rejects an incompatible child before readiness" {
  try
    prepare(
      @worker_executable_fixture.cat_path(),
      supervisor_startup(),
      supervisor_source(),
      supervisor_process_limits(),
    )
  catch {
    Invalid(Submission) => ()
    _ => fail("unexpected incompatible-child handshake error")
  } noraise {
    _ => fail("unhandshaken child became ready")
  }
}

///|
test "startup source mismatch fails before executable spawn" {
  try
    prepare(
      @worker_executable_fixture.cat_path(),
      supervisor_startup(),
      alternate_supervisor_source(),
      supervisor_process_limits(),
    )
  catch {
    Invalid(SourceMismatch) => ()
    _ => fail("unexpected source mismatch error")
  } noraise {
    _ => fail("foreign bootstrap source reached spawn")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "supervisor retains exact immutable bootstrap source bytes" {
  let owner = test_supervisor()
  let expected = supervisor_source()
  let retained = owner.bootstrap_source()
  assert_eq(retained.digest(), expected.digest())
  assert_eq(retained.length(), expected.length())
  let left = FixedArray::make(expected.capacity(), b'\x00')
  let right = FixedArray::make(retained.capacity(), b'\x00')
  expected.copy_to(left, destination_offset=0)
  retained.copy_to(right, destination_offset=0)
  for index in 0..<expected.length() {
    assert_eq(left[index], right[index])
  }
  owner.close()
}

///|
test "bootstrap source capacity rejects only undersized buffers before child spawn" {
  let base = supervisor_startup()
  let tiny_worker = @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=1,
    max_capabilities=1,
    max_completion_slots=1,
    max_sequence_tokens=1,
    max_token_id=0,
  )
  let tiny_inference = @inference.InferenceLimits::new(
    max_text_bytes=1,
    max_input_tokens=1,
    max_new_tokens=1,
    max_context_tokens=1,
    max_token_id=0,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1UL,
    max_top_k=1,
    max_temperature=1.0,
  )
  let startup = @worker_wire.WorkerStartupContract::new(
    identity=base.identity(),
    bootstrap_digest=base.bootstrap_digest(),
    bootstrap_source_digest=base.bootstrap_source_digest(),
    device_ordinal=base.device_ordinal(),
    model_generation_value=base.model_generation_value(),
    worker_limits=tiny_worker,
    inference_limits=tiny_inference,
    has_predecessor=false,
    predecessor_value=0UL,
  )
  let required = @worker_wire.bootstrap_source_capacity()
  for bytes in [required - 1, required] {
    let limits = WorkerProcessLimits::new(
      max_frame_bytes=bytes,
      startup_io_timeout_millis=1,
      io_timeout_millis=1,
      shutdown_timeout_millis=1,
    )
    try
      prepare(
        @worker_executable_fixture.cat_path(),
        startup,
        supervisor_source(),
        limits,
      )
    catch {
      Invalid(Capacity) if bytes == required - 1 => ()
      Invalid(Capacity) => fail("exact bootstrap source capacity was rejected")
      _ if bytes == required => ()
      _ => fail("unexpected startup capacity result")
    } noraise {
      _ => fail("incompatible cat child unexpectedly became ready")
    }
  }
}

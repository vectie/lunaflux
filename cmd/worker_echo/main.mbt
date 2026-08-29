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
    raise Failure("worker startup kind mismatch")
  }
  let startup = configured.contract()
  let source_input = FixedArray::make(
    @worker_wire.bootstrap_source_capacity(),
    b'\x00',
  )
  let source_length = channel.read_frame(source_input)
  let source = @worker_wire.decode_bootstrap_source(source_input, source_length)
  if source.digest() != startup.bootstrap_source_digest() {
    raise Failure("worker bootstrap source mismatch")
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
  let graph_output = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let mut graph_misses = 0UL
  let mut running = true
  while running {
    let length = channel.read_frame_or_eof(input) catch {
      _ => raise Failure("worker frame receive failed")
    }
    if length == 0 {
      running = false
    }
    if running {
      let frame = plan_owner.load(input, length)
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
      if graph_misses != 0xffffffffffffffffUL {
        graph_misses += 1UL
      }
      let graph = @worker_wire.WorkerGraphTelemetry::without_shape(
        sequence=completion.sequence_value(),
        path=WorkerGraphOrderedEager,
        graph_hits=0UL,
        graph_misses~,
      )
      let graph_length = @worker_wire.encode_worker_graph_telemetry(
        graph, graph_output,
      )
      channel.write_frame(graph_output, offset=0, byte_count=graph_length)
    }
  }
}

///|
fn main {
  try! run_worker()
}

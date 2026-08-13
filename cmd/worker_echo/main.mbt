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
  let mut running = true
  while running {
    let length = try channel.read_frame(input) catch {
      ChannelClosed(Read) => {
        running = false
        0
      }
      _ => raise Failure("worker frame receive failed")
    } noraise {
      value => value
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
    }
  }
}

///|
fn main {
  try! run_worker()
}

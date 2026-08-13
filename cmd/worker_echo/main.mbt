///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=8,
    max_plan_tokens=32,
    max_plan_pages=32,
    max_capabilities=32,
    max_completion_slots=8,
    max_sequence_tokens=64,
    max_token_id=1024,
  )
}

///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=64,
    max_new_tokens=64,
    max_context_tokens=64,
    max_token_id=1024,
    max_stop_token_ids=8,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=64,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=64,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn run_worker() -> Unit raise {
  let limits = worker_limits()
  let inference = inference_limits()
  let channel = @process.open_inherited(
    @process.ProcessLimits::new(
      max_frame_bytes=4096,
      io_timeout_millis=5000,
      shutdown_timeout_millis=5000,
    ),
  )
  let plan_owner = @worker_wire.PlanFrameBuffer::new(limits, inference)
  let completion_owner = @worker_wire.CompletionFrameBuffer::new(
    limits, inference,
  )
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

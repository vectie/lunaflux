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
  let configured = startup_owner.load(
    startup_input,
    channel.read_frame(startup_input),
  )
  if configured.kind() != Configure {
    raise Failure("worker startup kind mismatch")
  }
  let startup = configured.contract()
  let source_input = FixedArray::make(
    @worker_wire.bootstrap_source_capacity(),
    b'\x00',
  )
  let source = @worker_wire.decode_bootstrap_source(
    source_input,
    channel.read_frame(source_input),
  )
  if source.digest() != startup.bootstrap_source_digest() {
    raise Failure("worker bootstrap source mismatch")
  }
  let ready = startup_owner.encode(Ready, startup)
  let startup_output = FixedArray::make(startup_owner.capacity(), b'\x00')
  ready.copy_to(startup_output, destination_offset=0)
  channel.write_frame(startup_output, offset=0, byte_count=ready.length())
  let plan_owner = @worker_wire.PlanFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  )
  let input = FixedArray::make(plan_owner.capacity(), b'\x00')
  ignore(channel.read_frame(input))
  // Exit after accepting the first physical plan without returning completion.
}

///|
fn main {
  try! run_worker()
}

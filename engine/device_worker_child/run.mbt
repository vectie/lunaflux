///|
const CHILD_MAX_FRAME_BYTES : Int = 67108864

///|
// The child does not receive the parent's deployment I/O policy. Use the
// process boundary's admitted maximum so a valid slower parent is never timed
// out first by hidden child policy; the parent retains its shorter deadline and
// kill/reap authority.
const CHILD_IO_TIMEOUT_MILLIS : Int = 600000

///|
fn read_configure(
  channel : @process.InheritedChannel,
  owner : @worker_wire.StartupFrameBuffer,
  input : FixedArray[Byte],
) -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError {
  let length = channel.read_frame(input) catch { _ => raise StartupFailed }
  let frame = owner.load(input, length) catch { _ => raise StartupFailed }
  let kind = frame.kind() catch { _ => raise StartupFailed }
  if kind != Configure {
    raise StartupFailed
  }
  frame.contract() catch {
    _ => raise StartupFailed
  }
}

///|
fn read_source(
  channel : @process.InheritedChannel,
  input : FixedArray[Byte],
) -> @worker_wire.EncodedBootstrapSource raise DeviceWorkerChildError {
  let length = channel.read_frame(input) catch { _ => raise StartupFailed }
  @worker_wire.decode_bootstrap_source(input, length) catch {
    _ => raise StartupFailed
  }
}

///|
fn prepare_steady_buffers(
  startup : @worker_wire.WorkerStartupContract,
) -> ChildSteadyBuffers raise DeviceWorkerChildError {
  let plan_owner = @worker_wire.PlanFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  ) catch {
    _ => raise StartupFailed
  }
  let completion_owner = @worker_wire.CompletionFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  ) catch {
    _ => raise StartupFailed
  }
  {
    plan_owner,
    completion_owner,
    input: FixedArray::make(plan_owner.capacity(), b'\x00'),
    output: FixedArray::make(completion_owner.capacity(), b'\x00'),
  }
}

///|
fn serve_frames(
  channel : @process.InheritedChannel,
  owner : @device_worker_bootstrap.DeviceWorkerBootstrapOwner,
  buffers : ChildSteadyBuffers,
) -> Unit raise DeviceWorkerChildError {
  serve_serialized_with(
    () => {
      let length = channel.read_frame_or_eof(buffers.input) catch {
        _ => raise StartupFailed
      }
      if length == 0 {
        raise CleanEof
      }
      buffers.plan_owner.load(buffers.input, length) catch {
        _ => raise StartupFailed
      }
    },
    plan => {
      buffers.completion_owner.writer_for(plan) catch {
        _ => raise StartupFailed
      }
    },
    (plan, writer) => {
      owner.execute_frame(plan, writer) catch {
        _ => raise StartupFailed
      }
    },
    completion => {
      let length = completion.length() catch { _ => raise StartupFailed }
      completion.copy_to(buffers.output, destination_offset=0) catch {
        _ => raise StartupFailed
      }
      channel.write_frame(buffers.output, offset=0, byte_count=length) catch {
        _ => raise StartupFailed
      }
    },
  )
}

///|
/// Runs the device child over the inherited private channel.
///
/// The command consumes exactly Configure then bootstrap-source frames, emits
/// Ready only after real bootstrap admission and an exact readiness query,
/// then serially validates and executes bounded plan frames until clean EOF.
/// `false` is payload-free failure evidence for the command's silent nonzero
/// exit boundary.
pub fn run() -> Bool {
  let limits = @process.ProcessLimits::new(
    max_frame_bytes=CHILD_MAX_FRAME_BYTES,
    io_timeout_millis=CHILD_IO_TIMEOUT_MILLIS,
    shutdown_timeout_millis=CHILD_IO_TIMEOUT_MILLIS,
  ) catch {
    _ => return false
  }
  let channel = @process.open_inherited(limits)
  let startup_owner = @worker_wire.StartupFrameBuffer::new()
  let startup_input = FixedArray::make(startup_owner.capacity(), b'\x00')
  let source_input = FixedArray::make(
    @worker_wire.bootstrap_source_capacity(),
    b'\x00',
  )
  let ready_output = FixedArray::make(startup_owner.capacity(), b'\x00')
  run_startup_with(
    () => read_configure(channel, startup_owner, startup_input),
    () => read_source(channel, source_input),
    (startup, source) => {
      let preparation = @device_worker_bootstrap.prepare(startup, source) catch {
        _ => raise StartupFailed
      }
      match preparation {
        ChildReady(owner) => BootstrapReady(owner)
        CleanupRequired(failed) => BootstrapCleanup(failed)
      }
    },
    failed => failed.retry_close() catch { _ => raise StartupFailed },
    owner => owner.readiness_contract() catch { _ => raise StartupFailed },
    startup => prepare_steady_buffers(startup),
    startup => {
      let ready = startup_owner.encode(Ready, startup) catch {
        _ => raise StartupFailed
      }
      let length = ready.length() catch { _ => raise StartupFailed }
      ready.copy_to(ready_output, destination_offset=0) catch {
        _ => raise StartupFailed
      }
      channel.write_frame(ready_output, offset=0, byte_count=length) catch {
        _ => raise StartupFailed
      }
    },
    (owner, buffers) => serve_frames(channel, owner, buffers),
    owner => owner.close() catch { _ => raise StartupFailed },
  )
}

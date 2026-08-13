///|
const CHILD_MAX_FRAME_BYTES : Int = 67108864

///|
const CHILD_IO_TIMEOUT_MILLIS : Int = 5000

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
fn wait_for_parent_eof(
  channel : @process.InheritedChannel,
) -> Unit raise DeviceWorkerChildError {
  channel.expect_clean_eof() catch {
    _ => raise StartupFailed
  }
}

///|
/// Runs the startup-only device child over the inherited private channel.
///
/// The command consumes exactly Configure then bootstrap-source frames, emits
/// Ready only after real bootstrap admission and an exact readiness query,
/// accepts no steady-state frame, and closes the device owner after parent EOF.
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
    () => wait_for_parent_eof(channel),
    owner => owner.close() catch { _ => raise StartupFailed },
  )
}

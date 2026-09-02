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
fn read_parent_approval_attestation(
  channel : @process.InheritedChannel,
  input : FixedArray[Byte],
) -> @luna_capability_manifest.LunaParentStartupApprovalWitness? raise DeviceWorkerChildError {
  let length = channel.read_frame(input) catch { _ => raise StartupFailed }
  @worker_wire.decode_parent_approval_attestation_v1(input, length) catch {
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
    graph_output: FixedArray::make(
      @worker_wire.worker_graph_telemetry_frame_capacity(),
      b'\x00',
    ),
  }
}

///|
fn child_graph_telemetry(
  sequence : UInt64,
  report : @device_step.PagedGraphRuntimeTelemetry,
) -> @worker_wire.WorkerGraphTelemetry raise DeviceWorkerChildError {
  let path = match report.selected_path() {
    PagedGraphCaptured => @worker_wire.WorkerGraphCaptured
    PagedGraphOrderedEager => WorkerGraphOrderedEager
  }
  let hits = report.graph_hits()
  let misses = report.graph_misses()
  match report.selected_shape_class() {
    None =>
      @worker_wire.WorkerGraphTelemetry::without_shape(
        sequence~,
        path~,
        graph_hits=hits,
        graph_misses=misses,
      ) catch {
        _ => raise StartupFailed
      }
    Some(shape) => {
      let wire_shape = @worker_wire.WorkerGraphShapeClass::new(
        phase=shape.phase().as_stable_int(),
        batch_rows=shape.batch_rows(),
        query_tokens_per_row=shape.query_tokens_per_row(),
        context_tokens=shape.context_tokens(),
        token_rows=shape.token_rows(),
      ) catch {
        _ => raise StartupFailed
      }
      @worker_wire.WorkerGraphTelemetry::with_shape(
        sequence~,
        path~,
        shape=wire_shape,
        graph_hits=hits,
        graph_misses=misses,
      ) catch {
        _ => raise StartupFailed
      }
    }
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
      let sequence = completion.sequence_value() catch {
        _ => raise StartupFailed
      }
      if @worker_wire.worker_graph_telemetry_sample_required(sequence) {
        let report = owner.graph_runtime_telemetry() catch {
          _ => raise StartupFailed
        }
        let graph_length = @worker_wire.encode_worker_graph_telemetry(
          child_graph_telemetry(sequence, report),
          buffers.graph_output,
        ) catch {
          _ => raise StartupFailed
        }
        channel.write_frame(
          buffers.graph_output,
          offset=0,
          byte_count=graph_length,
        ) catch {
          _ => raise StartupFailed
        }
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
  let attestation_input = FixedArray::make(
    @worker_wire.parent_approval_attestation_capacity(),
    b'\x00',
  )
  let mut parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness? = None
  let ready_output = FixedArray::make(startup_owner.capacity(), b'\x00')
  run_startup_with(
    () => read_configure(channel, startup_owner, startup_input),
    () => {
      let source = read_source(channel, source_input)
      parent_witness = read_parent_approval_attestation(
        channel, attestation_input,
      )
      source
    },
    (startup, source) => {
      let preparation = @device_worker_bootstrap.prepare(
        startup,
        source,
        luna_parent_witness=parent_witness,
      ) catch {
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

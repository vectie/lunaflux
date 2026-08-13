///|
fn child_source(
  config_digest : String,
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/labels-are-not-authority/model",
      kernel_root="/labels-are-not-authority/kernel",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.safetensors",
      config_digest_sha256=config_digest,
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="execution.json",
      manifest_digest_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
      tokens_per_page=4,
      total_page_count=8,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=4096,
      max_weight_file_bytes=4096L,
      max_weight_arena_bytes=4096L,
      max_activation_arena_bytes=65536L,
      max_kv_arena_bytes=65536L,
      max_execution_manifest_bytes=65536L,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
    ),
  )
}

///|
fn child_startup(
  source : @worker_wire.EncodedBootstrapSource,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
      @spec.PlanDigest::from_sha256(
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      ),
    ),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    ),
    bootstrap_source_digest=source.digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=@worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=1,
      max_decode_rows=1,
      max_plan_rows=1,
      max_plan_tokens=4,
      max_plan_pages=2,
      max_capabilities=64,
      max_completion_slots=1,
      max_sequence_tokens=16,
      max_token_id=7,
    ),
    inference_limits=@inference.InferenceLimits::new(
      max_text_bytes=32,
      max_input_tokens=16,
      max_new_tokens=16,
      max_context_tokens=16,
      max_token_id=7,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=8,
      max_trace_bytes=8,
      max_cache_scope_bytes=8,
      max_decoded_delta_bytes=8,
      max_deadline_millis=100UL,
      max_top_k=8,
      max_temperature=4.0,
    ),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn process_limits() -> @process.ProcessLimits raise {
  @process.ProcessLimits::new(
    max_frame_bytes=67108864,
    io_timeout_millis=5000,
    shutdown_timeout_millis=5000,
  )
}

///|
fn send_startup(
  child : @process.ChildProcess,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> Unit raise {
  let startup_owner = @worker_wire.StartupFrameBuffer::new()
  let startup_output = FixedArray::make(startup_owner.capacity(), b'\x00')
  let configure = startup_owner.encode(Configure, startup)
  configure.copy_to(startup_output, destination_offset=0)
  child.write_frame(startup_output, offset=0, byte_count=configure.length())
  let source_output = FixedArray::make(source.length(), b'\x00')
  source.copy_to(source_output, destination_offset=0)
  child.write_frame(source_output, offset=0, byte_count=source.length())
}

///|
fn assert_no_ready_nonzero(child : @process.ChildProcess) -> Unit raise {
  let response = FixedArray::make(
    @worker_wire.StartupFrameBuffer::new().capacity(),
    b'\x00',
  )
  try child.read_frame(response) catch {
    ChannelClosed(Read) => ()
    _ => fail("unexpected rejected-child channel failure")
  } noraise {
    _ => fail("rejected child published Ready")
  }
  let exit = child.wait()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 1)
  child.close()
}

///|
async fn run(executable : Bytes) -> Unit {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-device-worker-child")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let model_path = "\{parent}/model"
    let kernel_path = "\{parent}/kernel"
    @fs.mkdir(model_path)
    @fs.mkdir(kernel_path)
    let invalid_config = b"{\"model_type\":\"not-llama\"}"
    @fs.write_file(
      "\{model_path}/config.json",
      invalid_config,
      create_mode=CreateNew,
    )
    let config_digest = @crypto.bytes_to_hex_string(
      @crypto.sha256(invalid_config),
    )
    let source = child_source(config_digest)
    let startup = child_startup(source)
    let model_root = @approved_fs.ApprovedRoot::open_absolute(model_path)
    let kernel_root = @approved_fs.ApprovedRoot::open_absolute(kernel_path)
    let roots = @approved_fs.acquire_worker_approved_roots(
      model_root~,
      kernel_root~,
    )
    model_root.close()
    kernel_root.close()
    let child = @process.spawn_with_approved_roots(
      executable,
      roots,
      process_limits(),
    )
    send_startup(child, startup, source)
    assert_no_ready_nonzero(child)

    let unexpected = @process.spawn_with_approved_roots(
      executable,
      roots,
      process_limits(),
    )
    let startup_owner = @worker_wire.StartupFrameBuffer::new()
    let configure = startup_owner.encode(Configure, startup)
    let configure_bytes = FixedArray::make(startup_owner.capacity(), b'\x00')
    configure.copy_to(configure_bytes, destination_offset=0)
    unexpected.write_frame(
      configure_bytes,
      offset=0,
      byte_count=configure.length(),
    )
    let invalid_source_frame = FixedArray::from_array([b'!'])
    unexpected.write_frame(invalid_source_frame, offset=0, byte_count=1)
    assert_no_ready_nonzero(unexpected)
    @approved_fs.close_worker_approved_roots(roots)
  }
}

///|
async fn main {
  match @env.args() {
    [_, executable] => run(@utf8.encode(executable))
    _ => fail("device child executable path is required")
  }
}

///|
fn approved_weight_file_limits() -> @device_materialize.DeviceWeightFileLimits raise {
  @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=210712L,
    transfer_chunk_bytes=4096,
    safetensors=approved_header_limits(),
  )
}

///|
fn approved_execution_limits(
  operation_count : Int,
) -> @execution_manifest_file.ExecutionManifestFileLimits raise {
  let artifact_limits = @artifact.KernelArtifactLimits::new(
    max_modules=32,
    max_entry_points=64,
    max_module_bytes=4194304L,
    max_total_module_bytes=67108864L,
    max_function_symbol_bytes=96,
  )
  @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=1048576L,
    max_json_depth=24,
    max_modules=32,
    max_operations=operation_count,
    device_memory=@device_memory.DeviceMemoryLimits::new(
      activation_alignment=16L,
      max_arena_bytes=33554432L,
    ),
    launch_contract=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=operation_count,
      max_operands_per_contract=16,
    ),
    artifact_file=@artifact_file.KernelArtifactFileLimits::new(
      max_manifest_bytes=1048576L,
      max_json_depth=24,
      max_path_bytes=128,
      artifact=artifact_limits,
    ),
  )
}

///|
fn approved_weight_layout(
  inputs : ApprovedBf16Inputs,
) -> @device_materialize.DeviceWeightLayout raise {
  @device_materialize.plan_layout(
    inputs.bindings,
    alignment_bytes=16L,
    max_arena_bytes=1048576L,
  )
}

///|
async fn write_approved_execution_manifest(
  runtime_root : String,
  release : @luna_bf16_kernel_producer.LunaBf16KernelReleaseInput,
) -> @execution_manifest_file.ExecutionManifestDigest {
  let manifest = release.bundle().execution_manifest()
  @fs.write_file(
    "\{runtime_root}/execution.json",
    manifest.bytes(),
    create_mode=CreateNew,
    permission=0o400,
  )
  @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    manifest.digest_sha256(),
  )
}

///|
fn close_failed_weights(
  failed : @device_materialize.FailedDeviceWeightPreparation,
) -> @device_materialize.DeviceWeights raise {
  let cause = failed.cause()
  failed.retry_close()
  raise cause
}

///|
fn close_failed_executor(
  failed : @device_step.FailedPagedGraphPreparation,
) -> @device_step.PagedGraphExecutor raise {
  let cause = failed.cause()
  failed.retry_close()
  raise cause
}

///|
async fn run_approved_bf16_physical(
  model_root_path : String,
  compiled_root : String,
  runtime_root : String,
  toolchain_sha256 : String,
  driver_identity_sha256 : String,
  compiler_major : UInt,
  compiler_minor : UInt,
  compiler_patch : UInt,
) -> Double {
  let inputs = load_approved_bf16_inputs(model_root_path)
  let corpus = approved_corpus_case(inputs.corpus)
  let candidates = approved_candidate_set(
    inputs, toolchain_sha256, compiler_major, compiler_minor, compiler_patch,
  )
  let compilations = approved_offline_compilations(
    compiled_root, candidates, toolchain_sha256, driver_identity_sha256,
  )
  let layout = approved_kv_layout(inputs.plan)
  let target = approved_target()
  let release = @luna_bf16_kernel_producer.produce_luna_bf16_kernel_release(
    inputs.plan,
    layout,
    target,
    compilations,
    approved_producer_limits(),
  )
  let runtime_canonical = @fs.realpath(runtime_root)
  if runtime_canonical != runtime_root {
    raise InvalidInput("runtime root must be canonical and absolute")
  }
  let manifest_digest = write_approved_execution_manifest(runtime_root, release)
  let model_root = @approved_fs.ApprovedRoot::open_absolute(model_root_path)
  defer (try! model_root.close())
  let kernel_root = @approved_fs.ApprovedRoot::open_absolute(runtime_root)
  defer (try! kernel_root.close())
  let inspection = @device_materialize.inspect_file(
    model_root,
    inputs.metadata,
    inputs.plan,
    @approved_fs.ApprovedRelativeLocator::new("model.safetensors"),
    limits=approved_weight_file_limits(),
    alignment_bytes=16L,
    max_arena_bytes=1048576L,
  )
  if inspection.layout() != approved_weight_layout(inputs) ||
    inspection.content_digest().as_hex() != PINNED_MODEL_SHA256 {
    raise InvalidInput("approved weight inspection identity mismatch")
  }
  let execution = @execution_manifest_file.load_paged(
    kernel_root,
    @approved_fs.ApprovedRelativeLocator::new("execution.json"),
    manifest_digest,
    inputs.plan,
    inspection.layout(),
    layout,
    target,
    approved_step_limits(),
    approved_execution_limits(inputs.plan.operations().length()),
  )
  let context = @device.open_context(0)
  defer (try! context.close())
  if !target.matches_capability(context.capability()) {
    raise InvalidInput("device 0 is not the exact approved sm120 BF16 target")
  }
  let weights = match
    @device_materialize.load_inspected_file(context, model_root, inspection) {
    WeightsReady(owner) => owner
    WeightsCleanupRequired(failed) => close_failed_weights(failed)
  }
  defer (try! weights.close())
  let executor = match
    @device_step.prepare_paged_graph_executor(
      context,
      inputs.plan,
      @worker_protocol.ModelPlanGeneration::first(),
      layout,
      approved_step_limits(),
      execution.static_plan(),
      execution.memory_plan(),
      execution.profile_id(),
      weights,
      execution.contracts(),
      execution.artifacts(),
      has_accepted_predecessor=false,
      accepted_predecessor_value=0UL,
      graph_policy=PagedCapturedRequired,
    ) {
    PagedReady(owner) => owner
    PagedCleanupRequired(failed) => close_failed_executor(failed)
  }
  defer (try! executor.close())
  execute_approved_continuation(executor, inputs.plan, corpus)
}

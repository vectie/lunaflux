///|
fn fixture_execution_limits(
  operation_count : Int,
) -> @execution_manifest_file.ExecutionManifestFileLimits raise {
  let artifact = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=operation_count,
    max_module_bytes=1024L,
    max_total_module_bytes=1024L,
    max_function_symbol_bytes=32,
  )
  @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=65536L,
    max_json_depth=16,
    max_modules=1,
    max_operations=operation_count,
    device_memory=@device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=65536L,
    ),
    launch_contract=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=operation_count,
      max_operands_per_contract=16,
    ),
    artifact_file=@artifact_file.KernelArtifactFileLimits::new(
      max_manifest_bytes=65536L,
      max_json_depth=16,
      max_path_bytes=64,
      artifact~,
    ),
  )
}

///|
fn fixture_bootstrap_limits() -> @device_step.DeviceBootstrapLimits raise {
  @device_step.DeviceBootstrapLimits::new(
    max_modules=1,
    max_entry_points=32,
    max_steps=32,
    max_operands_per_step=16,
    max_symbol_bytes=64,
    max_canonical_bytes=16384,
  )
}

///|
fn fixture_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=FIXTURE_MAX_ROWS,
    max_decode_rows=FIXTURE_MAX_ROWS,
    max_plan_rows=FIXTURE_MAX_ROWS,
    max_plan_tokens=FIXTURE_MAX_TOKENS,
    max_plan_pages=FIXTURE_MAX_PAGES,
    max_capabilities=FIXTURE_MAX_CAPABILITIES,
    max_completion_slots=FIXTURE_MAX_ROWS,
    max_sequence_tokens=FIXTURE_MAX_SEQUENCE_TOKENS,
    max_token_id=7,
  )
}

///|
fn fixture_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=7,
    max_stop_token_ids=2,
    max_stop_strings=1,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=100UL,
    max_top_k=8,
    max_temperature=4.0,
  )
}

///|
fn fixture_startup(
  execution : @execution_manifest_file.PagedExecutionAdmission,
  source_digest? : String = "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
) -> @worker_wire.WorkerStartupContract raise {
  let manifest = @device_step.admit_device_worker_bootstrap(
    execution.blueprint(),
    execution.artifacts(),
    device_ordinal=0,
    fixture_bootstrap_limits(),
  )
  manifest.worker_startup_contract(
    @worker_protocol.ModelPlanGeneration::first(),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      source_digest,
    ),
    fixture_worker_limits(),
    fixture_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
async fn write_fixture_files(
  directory : String,
  weights : Bytes,
  manifest : String,
  module_bytes : Bytes,
) -> Unit {
  @fs.write_file(
    "\{directory}/weights.safetensors",
    weights,
    create_mode=CreateNew,
  )
  @fs.write_file(
    "\{directory}/execution.json",
    @utf8.encode(manifest),
    create_mode=CreateNew,
  )
  @fs.write_file(
    "\{directory}/kernel.cubin",
    module_bytes,
    create_mode=CreateNew,
  )
}

///|
fn prepare_fixture_owner(
  directory : String,
  manifest : String,
  model : @spec.LlamaModelMetadata,
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
) -> @device_worker.DeviceWorkerOwner raise {
  let root = @approved_fs.ApprovedRoot::open_absolute(directory)
  let weight_limits = @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes=23,
    safetensors=@safetensors.HeaderLimits::new(
      max_header_bytes=8192,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=4096L,
      max_materialized_tensor_bytes=4096L,
    ),
  )
  let inspection = @device_materialize.inspect_file(
    root,
    model,
    model_plan,
    @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
    limits=weight_limits,
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
  let kv_layout = @device_layout.plan(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=64,
      max_arena_bytes=1048576L,
    ),
  )
  let step_limits = @device_step.DeviceStepLimits::new(
    max_rows=FIXTURE_MAX_ROWS,
    max_tokens=FIXTURE_MAX_TOKENS,
    max_pages=FIXTURE_MAX_PAGES,
    max_sequence_tokens=FIXTURE_MAX_SEQUENCE_TOKENS,
    max_token_id=7,
  )
  let execution = @execution_manifest_file.load_paged(
    root,
    @approved_fs.ApprovedRelativeLocator::new("execution.json"),
    @execution_manifest_file.ExecutionManifestDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(manifest))),
    ),
    model_plan,
    inspection.layout(),
    kv_layout,
    target,
    step_limits,
    fixture_execution_limits(model_plan.operations().length()),
  )
  let startup = fixture_startup(execution)
  let admitted = @device_worker.admit_plan(
    startup~,
    model~,
    weight_inspection=inspection,
    execution~,
    bootstrap_limits=fixture_bootstrap_limits(),
  )
  let drifted = fixture_startup(
    execution,
    source_digest="212233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
  )
  try @device_worker.prepare(drifted, root, admitted) catch {
    Invalid(StartupContract) => ()
    _ => {
      try! root.close()
      abort("unexpected bootstrap-source digest drift failure")
    }
  } noraise {
    unexpected => {
      match unexpected {
        DeviceWorkerReady(owner) => try! owner.close()
        DeviceWorkerCleanupRequired(failed) => try! failed.retry_close()
      }
      try! root.close()
      abort("bootstrap-source digest drift was admitted")
    }
  }
  let prepare_result : Result[
    @device_worker.DeviceWorkerPreparation,
    @device_worker.DeviceWorkerError,
  ] = Ok(@device_worker.prepare(startup, root, admitted)) catch {
    error => Err(error)
  }
  let prepared = match prepare_result {
    Ok(value) => value
    Err(error) => {
      try! root.close()
      raise error
    }
  }
  match prepared {
    DeviceWorkerReady(owner) => {
      let close_result : Result[Unit, @approved_fs.ApprovedFsError] = Ok(
        root.close(),
      ) catch {
        error => Err(error)
      }
      match close_result {
        Ok(_) => ()
        Err(error) => {
          try! owner.close()
          raise error
        }
      }
      owner
    }
    DeviceWorkerCleanupRequired(failed) => {
      let close_result : Result[Unit, @approved_fs.ApprovedFsError] = Ok(
        root.close(),
      ) catch {
        error => Err(error)
      }
      match close_result {
        Ok(_) => ()
        Err(error) => {
          try! failed.retry_close()
          raise error
        }
      }
      try! failed.retry_close()
      abort("fixture device worker preparation required cleanup")
    }
  }
}

///|
fn admit_i8_source_v6(
  model_root_label : StringView,
  kernel_root_label : StringView,
  claims : I8RuntimeDescriptorClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @worker_wire.EncodedBootstrapSource raise RuntimeDescriptorFileError {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root=model_root_label,
    kernel_root=kernel_root_label,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let model = @worker_wire.WorkerI8BootstrapModelSource::new(
    config_locator=claims.model.config_locator,
    numeric_weights_locator=claims.model.numeric_weights_locator,
    config_digest_sha256=claims.model.config_digest,
    numeric_weight_artifact_sha256=claims.model.numeric_weight_artifact_digest,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator=claims.kernels.manifest_locator,
    manifest_digest_sha256=claims.kernels.manifest_digest,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let execution = @worker_wire.WorkerI8BootstrapExecution::new(
    compute_major=claims.execution.compute_major,
    compute_minor=claims.execution.compute_minor,
    total_memory_bytes=claims.execution.total_memory_bytes,
    supports_bf16=claims.execution.supports_bf16,
    supports_cublas_lt=claims.execution.supports_cublas_lt,
    tokens_per_page=claims.execution.tokens_per_page,
    total_page_count=claims.execution.total_page_count,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  @worker_wire.admit_i8_bootstrap_source_v6(
    roots, model, kernel, execution, ceilings,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
}

///|
fn admit_i8_startup_contract(
  execution : @execution_manifest_file.I8PagedExecutionAdmissionV4,
  source : @worker_wire.EncodedBootstrapSource,
  generation_value : UInt64,
  worker : @worker_protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
) -> @worker_wire.WorkerStartupContract raise RuntimeDescriptorFileError {
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    generation_value,
  ) catch {
    _ => raise AdmissionRejected(StartupContract)
  }
  execution
  .bootstrap_manifest()
  .worker_startup_contract(
    generation,
    bootstrap_source_digest=source.digest(),
    worker,
    inference,
    has_predecessor=false,
    predecessor_value=0UL,
  ) catch {
    _ => raise AdmissionRejected(StartupContract)
  }
}

///|
fn admit_source(
  model_root_label : StringView,
  kernel_root_label : StringView,
  claims : RuntimeDescriptorClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval?,
) -> @worker_wire.EncodedBootstrapSource raise RuntimeDescriptorFileError {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root=model_root_label,
    kernel_root=kernel_root_label,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let model = @worker_wire.WorkerBootstrapModelSource::new(
    config_locator=claims.model.config_locator,
    weights_locator=claims.model.weights_locator,
    config_digest_sha256=claims.model.config_digest,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator=claims.kernels.manifest_locator,
    manifest_digest_sha256=claims.kernels.manifest_digest,
    luna_approval~,
    fused_runtime=match claims.kernels.fused_runtime {
      Some(fused) =>
        Some(
          @worker_wire.WorkerBootstrapFusedRuntimeSource::new(
            locator=fused.locator,
            digest_sha256=fused.digest,
          ),
        )
      None => None
    },
    sampling_runtime=claims.kernels.sampling_runtime,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let execution = @worker_wire.WorkerBootstrapExecution::new(
    compute_major=claims.execution.compute_major,
    compute_minor=claims.execution.compute_minor,
    supports_bf16=claims.execution.supports_bf16,
    supports_cublas_lt=claims.execution.supports_cublas_lt,
    tokens_per_page=claims.execution.tokens_per_page,
    total_page_count=claims.execution.total_page_count,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  @worker_wire.admit_bootstrap_source(roots, model, kernel, execution, ceilings) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
}

///|
fn admit_startup_contract(
  manifest : @device_step.DeviceWorkerBootstrapManifest,
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
  manifest.worker_startup_contract(
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

///|
fn tensor_parallel_weight_limits(
  plan : @plan.ModelPlan,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @device_materialize.DeviceWeightFileLimits raise RuntimeDescriptorFileError {
  let safetensors = @safetensors.HeaderLimits::new(
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    max_tensor_count=plan.tensor_count(),
    max_rank=RECIPE_MAX_RANK,
    max_name_length=RECIPE_MAX_TENSOR_NAME_BYTES,
    max_payload_bytes=ceilings.max_weight_file_bytes(),
    max_materialized_tensor_bytes=ceilings
      .max_weight_file_bytes()
      .min(ceilings.max_weight_arena_bytes()),
  ) catch {
    _ => raise AdmissionRejected(TensorParallelInspection)
  }
  @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    safetensors~,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelInspection)
  }
}

///|
fn admit_tensor_parallel_source(
  model_root_label : StringView,
  kernel_root_label : StringView,
  claims : TensorParallelRuntimeDescriptorClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval,
) -> @worker_wire.EncodedBootstrapSource raise RuntimeDescriptorFileError {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root=model_root_label,
    kernel_root=kernel_root_label,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
  let model = @worker_wire.WorkerBootstrapModelSource::new(
    config_locator=claims.model.config_locator,
    weights_locator=claims.model.weights_locator,
    config_digest_sha256=claims.model.config_digest,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator=claims.kernels.manifest_locator,
    manifest_digest_sha256=claims.kernels.manifest_digest,
    luna_approval=Some(luna_approval),
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
  let execution = @worker_wire.WorkerBootstrapExecution::new(
    compute_major=claims.execution.compute_major,
    compute_minor=claims.execution.compute_minor,
    supports_bf16=claims.execution.supports_bf16,
    supports_cublas_lt=claims.execution.supports_cublas_lt,
    tokens_per_page=claims.execution.tokens_per_page,
    total_page_count=claims.execution.total_page_count,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
  let ranks : Array[@worker_wire.TensorParallelRankRuntimeClaim] = []
  for claim in claims.tensor_parallel.ranks {
    ranks.push(
      @worker_wire.TensorParallelRankRuntimeClaim::new(
        rank=claim.rank,
        device_ordinal=claim.device_ordinal,
        memory_ceiling_bytes=claim.memory_ceiling_bytes,
        weight_reserved_bytes=claim.weight_reserved_bytes,
        activation_workspace_reserved_bytes=claim.activation_workspace_reserved_bytes,
        kv_reserved_bytes=claim.kv_reserved_bytes,
        collective_reserved_bytes=claim.collective_reserved_bytes,
      ) catch {
        _ => raise AdmissionRejected(TensorParallelSource)
      },
    )
  }
  let policy = @worker_wire.TensorParallelBootstrapPolicy::new(
    ranks,
    minimum_collective_runtime_version_code=claims.tensor_parallel.minimum_collective_runtime_version_code,
    maximum_collective_runtime_version_code=claims.tensor_parallel.maximum_collective_runtime_version_code,
    collective_exchange_timeout_millis=claims.tensor_parallel.collective_exchange_timeout_millis,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
  @worker_wire.admit_tensor_parallel_approved_bootstrap_source(
    roots, model, kernel, execution, ceilings, policy,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
}

///|
fn load_tensor_parallel_bound(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval,
  luna_verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> TensorParallelRuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  let bytes = read_descriptor_snapshot(
    model_root, descriptor_locator, expected_digest, limits,
  )
  let claims = parse_tensor_parallel_descriptor(bytes, limits)
  if claims.kernels.policy != "deployment_approved_aot_only" ||
    claims.execution.device_ordinal !=
    claims.tensor_parallel.ranks[0].device_ordinal {
    raise AdmissionRejected(TensorParallelSource)
  }
  let ceilings = admit_ceilings(claims.ceilings)
  if luna_verifier is Some(verifier) {
    if luna_approval.manifest_digest() != claims.kernels.manifest_digest ||
      luna_approval.approved_source() != claims.kernels.manifest_digest {
      raise AdmissionRejected(ExecutionManifest)
    }
    let manifest_locator = @approved_fs.ApprovedRelativeLocator::new(
      claims.kernels.manifest_locator,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    }
    let manifest_bytes = read_execution_manifest_snapshot(
      kernel_root,
      manifest_locator,
      claims.kernels.manifest_digest,
      ceilings.max_execution_manifest_bytes(),
    )
    ignore(
      verifier.authenticate_manifest_claim(
        manifest_bytes,
        manifest_digest_sha256=luna_approval.manifest_digest(),
        key_identity_sha256=luna_approval.key_identity(),
        signature_digest_sha256=luna_approval.signature_digest(),
        approved_source_sha256=luna_approval.approved_source(),
      ) catch {
        _ => raise AdmissionRejected(ExecutionManifest)
      },
    )
  }
  let model = admit_model(model_root, claims.model, ceilings)
  let plan = admit_plan(model, claims.model.max_batch_rows)
  let kv_layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=claims.execution.tokens_per_page,
      total_page_count=claims.execution.total_page_count,
      max_arena_bytes=ceilings.max_kv_arena_bytes(),
    ) catch {
      _ => raise AdmissionRejected(TensorParallelSource)
    },
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
  let worker = admit_worker_limits(claims.worker)
  let inference = admit_inference_limits(claims.inference)
  let target = admit_target(claims.execution)
  let ordinals : Array[Int] = []
  for rank in claims.tensor_parallel.ranks {
    ordinals.push(rank.device_ordinal)
  }
  let topology_selection = @device_topology.admit_rank_ordered_local_topology(
    ordinals, target,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelTopology)
  }
  let source = admit_tensor_parallel_source(
    model_root_label, kernel_root_label, claims, ceilings, luna_approval,
  )
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    claims.model.weights_locator,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelInspection)
  }
  let weight_limits = tensor_parallel_weight_limits(plan, ceilings)
  let inspections : Array[@device_materialize.DeviceShardedWeightFileInspection] = []
  for rank in 0..<ordinals.length() {
    inspections.push(
      @device_materialize.inspect_sharded_file(
        model_root,
        model,
        plan,
        locator,
        rank~,
        world_size=ordinals.length(),
        alignment_bytes=RECIPE_ALIGNMENT_BYTES,
        max_arena_bytes=ceilings.max_weight_arena_bytes(),
        limits=weight_limits,
      ) catch {
        _ => raise AdmissionRejected(TensorParallelInspection)
      },
    )
  }
  let operation_count = plan.operations().length()
  let manifest_limits = admit_execution_limits(operation_count, ceilings)
  let artifact_limits = manifest_limits.artifact_file().artifact()
  let device_step_limits = admit_step_limits(worker)
  let launch_limits = @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
    max_profiles=1,
    max_contracts=operation_count,
    max_operands_per_contract=RECIPE_MAX_OPERANDS,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
  let execution_limits = @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits::new(
    max_operations=operation_count,
    max_operands=operation_count * RECIPE_MAX_OPERANDS,
    max_descriptors=operation_count * RECIPE_MAX_OPERANDS,
    max_collectives=operation_count,
    max_descriptor_bytes=ceilings.max_activation_arena_bytes(),
    max_activation_workspace_bytes=ceilings.max_activation_arena_bytes(),
    artifact_limits~,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
  {
    descriptor_digest: expected_digest,
    model,
    plan,
    kv_layout,
    source,
    model_generation: @worker_protocol.ModelPlanGeneration::from_wire(
      claims.execution.model_generation,
    ) catch {
      _ => raise AdmissionRejected(StartupContract)
    },
    topology_selection,
    inspections: ReadOnlyArray::from_array(inspections),
    worker,
    inference,
    device_step_limits,
    manifest_limits,
    launch_limits,
    execution_limits,
    collective_exchange_timeout_millis: claims.tensor_parallel.collective_exchange_timeout_millis,
  }
}

///|
/// Loads one digest-pinned tensor-parallel descriptor and admits its exact
/// rank order against the local physical inventory before any context,
/// allocation, collective, child process, or listener is created.
pub fn load_tensor_parallel(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval,
  luna_verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff,
) -> TensorParallelRuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  bind_root_label(
    model_root,
    model_root_label,
    BindModelRoot,
    limits.max_descriptor_bytes(),
  )
  bind_root_label(
    kernel_root,
    kernel_root_label,
    BindKernelRoot,
    limits.max_descriptor_bytes(),
  )
  load_tensor_parallel_bound(
    model_root,
    kernel_root,
    model_root_label,
    kernel_root_label,
    descriptor_locator,
    expected_digest,
    limits,
    luna_approval,
    luna_verifier=Some(luna_verifier),
  )
}

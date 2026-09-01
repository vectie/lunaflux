///|
fn exact_startup_contract(
  startup : @worker_wire.WorkerStartupContract,
  manifest : @device_step.DeviceWorkerBootstrapManifest,
) -> @worker_wire.WorkerStartupContract raise DeviceWorkerError {
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  ) catch {
    error => raise Protocol(error)
  }
  let expected = manifest.worker_startup_contract(
    generation,
    bootstrap_source_digest=startup.bootstrap_source_digest(),
    startup.worker_limits(),
    startup.inference_limits(),
    has_predecessor=startup.has_predecessor(),
    predecessor_value=startup.predecessor_value(),
  ) catch {
    error => raise Bootstrap(error)
  }
  if expected != startup {
    raise Invalid(StartupContract)
  }
  expected
}

///|
fn validate_received_startup(
  received : @worker_wire.WorkerStartupContract,
  expected : @worker_wire.WorkerStartupContract,
) -> Unit raise DeviceWorkerError {
  if received != expected {
    raise Invalid(StartupContract)
  }
}

///|
fn validate_authenticated_model_plan(
  model : @spec.LlamaModelMetadata,
  startup : @worker_wire.WorkerStartupContract,
  execution : @execution_manifest_file.PagedExecutionAdmission,
) -> Unit raise DeviceWorkerError {
  let retained = execution.model_plan()
  let identity = retained.identity()
  if model.content_digest() != identity.content() ||
    startup.identity() != identity ||
    execution.blueprint().identity() != identity ||
    execution.weight_layout().identity() != identity ||
    execution.layout().identity() != identity {
    raise Invalid(ModelIdentity)
  }
}

///|
fn validate_weight_evidence(
  model : @spec.LlamaModelMetadata,
  inspection : @device_materialize.DeviceWeightFileInspection,
  execution : @execution_manifest_file.PagedExecutionAdmission,
) -> Unit raise DeviceWorkerError {
  let inspected_layout = inspection.layout()
  let execution_layout = execution.weight_layout()
  if inspection.content_digest() != model.content_digest() ||
    inspected_layout != execution_layout {
    raise Invalid(WeightEnvelope)
  }
}

///|
fn admit_optional_fused_residual_spans(
  model_plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : @device_step.PagedKvExecutionBlueprint,
  fused_catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog?,
  fused_bundle : @luna_fused_artifact_admission.ApprovedFusedKernelBundle?,
  reusable_fused_residual_runtime : @luna_fused_artifact_admission.ApprovedReusableFusedResidualRuntime?,
) -> ReadOnlyArray[@device_step.ApprovedFusedResidualRmsNormSpan] raise DeviceWorkerError {
  match (fused_catalog, fused_bundle, reusable_fused_residual_runtime) {
    (None, None, None) => ReadOnlyArray::from_array([])
    (Some(catalog), Some(bundle), None) =>
      @device_step.admit_production_fused_residual_rmsnorm_spans(
        model_plan, generation, blueprint, catalog, bundle,
      ) catch {
        _ => raise Invalid(ResourceState)
      }
    (None, None, Some(runtime)) =>
      @device_step.admit_reusable_fused_residual_rmsnorm_spans(
        model_plan, generation, blueprint, runtime,
      ) catch {
        _ => raise Invalid(ResourceState)
      }
    _ => raise Invalid(ResourceState)
  }
}

///|
/// Admits independently prepared immutable device-worker evidence without
/// opening a filesystem, native, or device resource. The exact weight-file
/// inspection and aggregate paged execution admission are retained privately.
pub fn admit_plan(
  startup~ : @worker_wire.WorkerStartupContract,
  model~ : @spec.LlamaModelMetadata,
  weight_inspection~ : @device_materialize.DeviceWeightFileInspection,
  execution~ : @execution_manifest_file.PagedExecutionAdmission,
  bootstrap_limits~ : @device_step.DeviceBootstrapLimits,
  sampling_policy? : @device_step.PagedSamplingExecutionPolicy = PagedHostSampling,
  fused_catalog? : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog? = None,
  fused_bundle? : @luna_fused_artifact_admission.ApprovedFusedKernelBundle? = None,
  reusable_fused_residual_runtime? : @luna_fused_artifact_admission.ApprovedReusableFusedResidualRuntime? = None,
  fused_qkv_readonly_artifacts? : @luna_paged_composite_artifact_admission.ApprovedFusedQkvReadOnlyProductionArtifacts? = None,
  fused_qkv_readonly_source? : @luna_paged_composite_artifact_admission.DigestPinnedFusedProductionRuntime? = None,
) -> DeviceWorkerPlan raise DeviceWorkerError {
  let blueprint = execution.blueprint()
  validate_authenticated_model_plan(model, startup, execution)
  validate_weight_evidence(model, weight_inspection, execution)
  @device_step.validate_paged_sampling_execution_policy(
    execution.model_plan(),
    blueprint,
    execution.artifacts(),
    sampling_policy,
  ) catch {
    _ => raise Invalid(ResourceState)
  }
  let manifest = @device_step.admit_device_worker_bootstrap(
    blueprint,
    execution.artifacts(),
    luna_authorization=execution.luna_authorization(),
    device_ordinal=startup.device_ordinal(),
    bootstrap_limits,
  ) catch {
    error => raise Bootstrap(error)
  }
  if manifest.identity() != execution.model_plan().identity() {
    raise Invalid(ModelIdentity)
  }
  let admitted_startup = exact_startup_contract(startup, manifest)
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    admitted_startup.model_generation_value(),
  ) catch {
    error => raise Protocol(error)
  }
  if fused_qkv_readonly_artifacts is Some(_) &&
    fused_qkv_readonly_source is Some(_) {
    raise Invalid(ResourceState)
  }
  let fused_residual_spans = admit_optional_fused_residual_spans(
    execution.model_plan(),
    generation,
    blueprint,
    fused_catalog,
    fused_bundle,
    reusable_fused_residual_runtime,
  )
  {
    startup: admitted_startup,
    model_plan: execution.model_plan(),
    weight_inspection: Some(weight_inspection),
    numeric_weight_inspection: None,
    execution,
    manifest,
    sampling_policy,
    fused_residual_spans,
    reusable_qwen_fused_attention_spans: ReadOnlyArray::from_array([]),
    qwen_decode_split_bindings: ReadOnlyArray::from_array([]),
    fused_qkv_readonly_artifacts,
    fused_qkv_readonly_source,
  }
}

///|
/// Admits the generic authenticated numeric-BF16 file path for the same dense
/// paged executor. No model-family type or scheduler policy crosses this seam.
pub fn admit_numeric_bf16_plan_v1(
  startup~ : @worker_wire.WorkerStartupContract,
  model_plan~ : @plan.ModelPlan,
  weight_inspection~ : @numeric_weight_file.NumericWeightFileInspection,
  execution~ : @execution_manifest_file.PagedExecutionAdmission,
  bootstrap_limits~ : @device_step.DeviceBootstrapLimits,
  sampling_policy? : @device_step.PagedSamplingExecutionPolicy = PagedHostSampling,
  fused_catalog? : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog? = None,
  fused_bundle? : @luna_fused_artifact_admission.ApprovedFusedKernelBundle? = None,
  reusable_fused_residual_runtime? : @luna_fused_artifact_admission.ApprovedReusableFusedResidualRuntime? = None,
  reusable_fused_runtime_bundle? : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle? = None,
  fused_qkv_readonly_artifacts? : @luna_paged_composite_artifact_admission.ApprovedFusedQkvReadOnlyProductionArtifacts? = None,
  fused_qkv_readonly_source? : @luna_paged_composite_artifact_admission.DigestPinnedFusedProductionRuntime? = None,
) -> DeviceWorkerPlan raise DeviceWorkerError {
  if execution.model_plan() != model_plan ||
    startup.identity() != model_plan.identity() ||
    weight_inspection.identity() != model_plan.identity() ||
    weight_inspection.numeric_schema_digest() !=
    model_plan.numeric_binding().digest() ||
    weight_inspection.layout() != execution.weight_layout() {
    raise Invalid(WeightEnvelope)
  }
  @device_step.validate_paged_sampling_execution_policy(
    model_plan,
    execution.blueprint(),
    execution.artifacts(),
    sampling_policy,
  ) catch {
    _ => raise Invalid(ResourceState)
  }
  let manifest = @device_step.admit_device_worker_bootstrap(
    execution.blueprint(),
    execution.artifacts(),
    luna_authorization=execution.luna_authorization(),
    device_ordinal=startup.device_ordinal(),
    bootstrap_limits,
  ) catch {
    error => raise Bootstrap(error)
  }
  let admitted_startup = exact_startup_contract(startup, manifest)
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    admitted_startup.model_generation_value(),
  ) catch {
    error => raise Protocol(error)
  }
  if fused_qkv_readonly_artifacts is Some(_) &&
    fused_qkv_readonly_source is Some(_) {
    raise Invalid(ResourceState)
  }
  let reusable_qwen_fused_runtime = admit_optional_reusable_qwen_runtime_bundle(
    model_plan,
    manifest.target(),
    reusable_fused_runtime_bundle,
    reusable_fused_residual_runtime,
    fused_catalog,
    fused_bundle,
    fused_qkv_readonly_artifacts,
    fused_qkv_readonly_source,
  )
  let fused_residual_spans = match reusable_qwen_fused_runtime {
    Some(runtime) if runtime.residual_rmsnorm_module() is Some(_) =>
      @device_step.admit_reusable_fused_residual_rmsnorm_bundle_spans(
        model_plan,
        generation,
        execution.blueprint(),
        runtime,
      ) catch {
        _ => raise Invalid(ResourceState)
      }
    _ =>
      admit_optional_fused_residual_spans(
        model_plan,
        generation,
        execution.blueprint(),
        fused_catalog,
        fused_bundle,
        reusable_fused_residual_runtime,
      )
  }
  let reusable_qwen_fused_attention_spans = match reusable_qwen_fused_runtime {
    Some(runtime) =>
      @device_step.admit_reusable_qwen_fused_attention_spans(
        model_plan,
        generation,
        execution.blueprint(),
        runtime,
      ) catch {
        _ => raise Invalid(ResourceState)
      }
    None => ReadOnlyArray::from_array([])
  }
  let qwen_decode_split_bindings = admit_reusable_qwen_decode_split_bindings(
    model_plan, execution, reusable_qwen_fused_runtime,
  )
  {
    startup: admitted_startup,
    model_plan,
    weight_inspection: None,
    numeric_weight_inspection: Some(weight_inspection),
    execution,
    manifest,
    sampling_policy,
    fused_residual_spans,
    reusable_qwen_fused_attention_spans,
    qwen_decode_split_bindings,
    fused_qkv_readonly_artifacts,
    fused_qkv_readonly_source,
  }
}

///|
/// Requires the received contract to equal the admitted inert plan before any
/// resource opens, then owns the exact configured device, fully re-admits and
/// streams the retained inspected descendant into final device storage, and
/// prepares the complete paged graph before returning readiness.
pub fn prepare(
  received : @worker_wire.WorkerStartupContract,
  model_root : @approved_fs.ApprovedRoot,
  plan : DeviceWorkerPlan,
) -> DeviceWorkerPreparation raise DeviceWorkerError {
  validate_received_startup(received, plan.startup)
  let ready_contract = plan.startup
  let execution = plan.execution
  let graph_policy = authenticated_bf16_graph_policy(execution)
  let resources : DeviceWorkerResources = {
    executor: None,
    weights: None,
    context: None,
    numeric_weights_owned_by_executor: false,
  }
  let context = @device.open_context(plan.manifest.device_ordinal()) catch {
    error => raise Device(error)
  }
  resources.context = Some(context)
  if context.ordinal() != plan.manifest.device_ordinal() {
    return preparation_failed(resources, Invalid(DeviceOrdinal))
  }
  if !plan.manifest.target().matches_capability(context.capability()) {
    return preparation_failed(resources, Invalid(DeviceTarget))
  }
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    plan.startup.model_generation_value(),
  ) catch {
    error => return preparation_failed(resources, Protocol(error))
  }
  let fused_qkv_readonly_artifacts = match
    (plan.fused_qkv_readonly_artifacts, plan.fused_qkv_readonly_source) {
    (Some(artifacts), None) => Some(artifacts)
    (None, Some(source)) =>
      Some(
        source.bind_live_device(context.device_identity()) catch {
          _ => return preparation_failed(resources, Invalid(ResourceState))
        },
      )
    (None, None) => None
    _ => return preparation_failed(resources, Invalid(ResourceState))
  }
  let fused_qkv_readonly_span = match fused_qkv_readonly_artifacts {
    Some(artifacts) =>
      Some(
        @device_step.admit_production_fused_qkv_readonly_attention_span(
          plan.model_plan,
          generation,
          execution.blueprint(),
          artifacts,
          context.device_identity(),
        ) catch {
          _ => return preparation_failed(resources, Invalid(ResourceState))
        },
      )
    None => None
  }
  let executor_outcome = match
    (plan.weight_inspection, plan.numeric_weight_inspection) {
    (Some(inspection), None) => {
      let weight_outcome = @device_materialize.load_inspected_file(
        context, model_root, inspection,
      ) catch {
        error => return preparation_failed(resources, Weights(error))
      }
      let weights = match weight_outcome {
        WeightsReady(owner) => {
          resources.weights = Some(Complete(owner))
          owner
        }
        WeightsCleanupRequired(failed) => {
          resources.weights = Some(Failed(failed))
          return preparation_failed(resources, Weights(failed.cause()))
        }
      }
      @device_step.prepare_paged_graph_executor(
        context,
        plan.model_plan,
        generation,
        execution.layout(),
        execution.device_step_limits(),
        execution.static_plan(),
        execution.memory_plan(),
        execution.profile_id(),
        weights,
        execution.contracts(),
        execution.artifacts(),
        luna_authorization=execution.luna_authorization(),
        graph_policy~,
        sampling_policy=plan.sampling_policy,
        fused_residual_spans=plan.fused_residual_spans,
        fused_qkv_readonly_span~,
        qwen_fused_attention_spans=plan.reusable_qwen_fused_attention_spans,
        decode_split_bindings=plan.qwen_decode_split_bindings,
        has_accepted_predecessor=plan.startup.has_predecessor(),
        accepted_predecessor_value=plan.startup.predecessor_value(),
      ) catch {
        error => return preparation_failed(resources, Executor(error))
      }
    }
    (None, Some(inspection)) => {
      let loaded = @numeric_weight_file.load_inspected_file(
        context, model_root, inspection,
      ) catch {
        error => return preparation_failed(resources, NumericWeights(error))
      }
      let numeric = match loaded {
        NumericWeightsReady(owner) => {
          resources.weights = Some(NumericComplete(owner))
          owner
        }
        NumericWeightsCleanupRequired(failed) => {
          resources.weights = Some(NumericFailed(failed))
          return preparation_failed(resources, NumericWeights(failed.cause()))
        }
      }
      let authority = numeric.authenticate(
        expected_identity=plan.model_plan.identity(),
        expected_schema_digest=plan.model_plan.numeric_binding().digest(),
        expected_layout=inspection.layout(),
        expected_artifact_digest=inspection.artifact_digest(),
      ) catch {
        error => return preparation_failed(resources, NumericWeights(error))
      }
      resources.weights = None
      resources.numeric_weights_owned_by_executor = true
      @device_step.prepare_numeric_bf16_paged_graph_executor_v1(
        context,
        plan.model_plan,
        generation,
        execution.layout(),
        execution.device_step_limits(),
        execution.static_plan(),
        execution.memory_plan(),
        execution.profile_id(),
        authority,
        execution.contracts(),
        execution.artifacts(),
        luna_authorization=execution.luna_authorization(),
        graph_policy~,
        sampling_policy=plan.sampling_policy,
        fused_residual_spans=plan.fused_residual_spans,
        fused_qkv_readonly_span~,
        qwen_fused_attention_spans=plan.reusable_qwen_fused_attention_spans,
        decode_split_bindings=plan.qwen_decode_split_bindings,
        has_accepted_predecessor=plan.startup.has_predecessor(),
        accepted_predecessor_value=plan.startup.predecessor_value(),
      ) catch {
        error => return preparation_failed(resources, Executor(error))
      }
    }
    _ => return preparation_failed(resources, Invalid(WeightEnvelope))
  }
  match executor_outcome {
    PagedReady(executor) => resources.executor = Some(Complete(executor))
    PagedCleanupRequired(failed) => {
      resources.executor = Some(Failed(failed))
      return preparation_failed(resources, Executor(failed.cause()))
    }
  }
  DeviceWorkerReady({
    startup: ready_contract,
    resources,
    lifecycle: WorkerReady,
  })
}

///|
/// Returns the exact configured contract only while every owned device/model/
/// executor resource is live. This is the sole readiness publication surface.
pub fn DeviceWorkerOwner::readiness_contract(
  self : DeviceWorkerOwner,
) -> @worker_wire.WorkerStartupContract raise DeviceWorkerError {
  if self.lifecycle != WorkerReady {
    raise InvalidLifecycle(self.lifecycle)
  }
  if !resources_complete(self.resources) {
    raise Invalid(ResourceState)
  }
  self.startup
}

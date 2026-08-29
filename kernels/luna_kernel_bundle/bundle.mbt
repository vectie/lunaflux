///|
fn compiled_operation_index(
  operations : ArrayView[LunaKernelCompiledOperation],
  operation_id : @plan.OperationId,
) -> Int? raise LunaKernelBundleError {
  let mut found : Int? = None
  for index, operation in operations {
    if operation.operation_id() == operation_id {
      if found is Some(_) {
        raise Invalid(DuplicateOperation)
      }
      found = Some(index)
    }
  }
  found
}

///|
fn module_index(
  modules : ArrayView[LunaKernelBundleModule],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, module_item in modules {
    if module_item.digest == digest {
      return Some(index)
    }
  }
  None
}

///|
fn insert_module(
  modules : Array[LunaKernelBundleModule],
  module_item : LunaKernelBundleModule,
) -> Unit {
  let mut insertion = modules.length()
  for index, existing in modules {
    if module_item.digest.sha256().compare(existing.digest.sha256()) < 0 {
      insertion = index
      break
    }
  }
  modules.insert(insertion, module_item)
}

///|
fn collect_exact_modules(
  operations : ArrayView[LunaKernelCompiledOperation],
  limits : LunaKernelBundleLimits,
) -> (Array[LunaKernelBundleModule], Int64) raise LunaKernelBundleError {
  let modules : Array[LunaKernelBundleModule] = []
  let mut total = 0L
  for operation in operations {
    let candidate = operation.module_item
    match module_index(modules, candidate.digest) {
      Some(index) => {
        let existing = modules[index]
        if existing.relative_path != candidate.relative_path ||
          existing.module_bytes != candidate.module_bytes {
          raise Invalid(DuplicateModule)
        }
      }
      None => {
        if modules.length() >= limits.max_modules {
          raise LimitExceeded(
            rule=DuplicateModule,
            maximum=limits.max_modules.to_int64(),
          )
        }
        let size = candidate.module_bytes.length().to_int64()
        if total > limits.max_total_module_bytes - size {
          raise LimitExceeded(
            rule=TotalModuleBytes,
            maximum=limits.max_total_module_bytes,
          )
        }
        total = total + size
        insert_module(modules, candidate)
      }
    }
  }
  (modules, total)
}

///|
/// Produces one canonical full-graph manifest and its exact content-addressed
/// CUBIN inventory. Input order is irrelevant; output operation order follows
/// the immutable model plan after unique full coverage is proved.
pub fn produce_luna_kernel_bundle(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  compiled_operations : ArrayView[LunaKernelCompiledOperation],
  limits : LunaKernelBundleLimits,
) -> LunaKernelBundleInput raise LunaKernelBundleError {
  let plan_operations = model_plan.operations()
  if plan_operations.length() > limits.max_operations ||
    compiled_operations.length() > limits.max_operations {
    raise LimitExceeded(
      rule=OperationCoverage,
      maximum=limits.max_operations.to_int64(),
    )
  }
  if compiled_operations.length() != plan_operations.length() {
    raise Invalid(OperationCoverage)
  }
  if contracts.model_identity() != model_plan.identity() {
    raise Invalid(ModelIdentity)
  }
  if contracts.device_target() != target {
    raise Invalid(DeviceTarget)
  }
  if contracts.catalog_version() != @catalog.CatalogVersion::v3() {
    raise Invalid(CatalogVersion)
  }
  let ordered : Array[LunaKernelCompiledOperation] = []
  let artifacts : Array[@full_graph_manifest.FullGraphAotOperationArtifact] = []
  for plan_operation in plan_operations {
    guard compiled_operation_index(compiled_operations, plan_operation.id())
      is Some(index) else {
      raise Invalid(OperationCoverage)
    }
    let compiled = compiled_operations[index]
    if compiled.target != target {
      raise Mismatch(
        rule=DeviceTarget,
        operation_index=plan_operation.id().as_int(),
      )
    }
    ordered.push(compiled)
    artifacts.push(compiled.artifact)
  }
  let manifest = @full_graph_manifest.produce_execution_manifest_v2(
    model_plan,
    target,
    contracts,
    artifacts,
    limits.manifest,
  ) catch {
    _ => raise Invalid(Manifest)
  }
  let (modules, total_module_bytes) = collect_exact_modules(ordered, limits)
  {
    manifest,
    modules: ReadOnlyArray::from_array(modules),
    operations: ReadOnlyArray::from_array(ordered),
    total_module_bytes,
  }
}

///|
/// All dense Llama-v1 semantic operations have a typed lowering family.
/// Auxiliary KV-cache-write is a pointwise side operation and is not a model
/// plan operation.
pub fn model_operation_has_typed_cuda_lowering(
  kind : @plan.OperationKind,
) -> Bool {
  match kind {
    Embedding | RmsNorm | QkRmsNorm | RotaryEmbedding | ResidualAdd => true
    QkvProjection | OutputProjection | GatedMlp | LanguageModelHead => true
    CausalAttention => true
  }
}

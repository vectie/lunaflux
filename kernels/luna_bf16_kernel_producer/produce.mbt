///|
fn exact_contract(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  operation_id : @plan.OperationId,
  profile : @launch_contract.PagedKernelProfile,
) -> @launch_contract.PagedKvAotLaunchContract raise LunaBf16KernelProducerError {
  let mut found : @launch_contract.PagedKvAotLaunchContract? = None
  for contract in contracts.contracts() {
    if contract.operation_id() == operation_id && contract.profile() == profile {
      if found is Some(_) {
        raise Invalid(LaunchContracts)
      }
      found = Some(contract)
    }
  }
  guard found is Some(contract) else { raise Invalid(LaunchContracts) }
  contract
}

///|
fn bind_compilation(
  prepared : PreparedCompilation,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  limits : @luna_kernel_bundle.LunaKernelBundleLimits,
) -> @luna_kernel_bundle.LunaKernelCompiledOperation raise LunaBf16KernelProducerError {
  let compilation = prepared.compilation
  let candidate = compilation.candidate
  let contract = exact_contract(
    contracts,
    prepared.operation.id(),
    candidate_profile(candidate),
  )
  match candidate {
    Pointwise(value) => {
      let bound = @luna_cuda_pointwise_aot.bind_manifest_pointwise_cuda_aot(
        value,
        prepared.operation,
        contracts,
      ) catch {
        _ => raise Invalid(FinalBinding)
      }
      @luna_kernel_bundle.admit_pointwise_compiled_operation(
        bound,
        contract,
        compilation.first_module,
        compilation.second_module,
        compilation.receipt,
        limits,
      ) catch {
        _ => raise Invalid(FinalBinding)
      }
    }
    Projection(value) => {
      let bound = @luna_cuda_projection_aot.bind_manifest_projection_cuda_aot(
        value,
        prepared.operation,
        contracts,
      ) catch {
        _ => raise Invalid(FinalBinding)
      }
      @luna_kernel_bundle.admit_projection_compiled_operation(
        bound,
        contract,
        compilation.first_module,
        compilation.second_module,
        compilation.receipt,
        limits,
      ) catch {
        _ => raise Invalid(FinalBinding)
      }
    }
    PagedAttention(value) => {
      let bound = @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_cuda_aot(
        value,
        prepared.operation,
        contracts,
      ) catch {
        _ => raise Invalid(FinalBinding)
      }
      @luna_kernel_bundle.admit_paged_attention_compiled_operation(
        bound,
        contract,
        compilation.first_module,
        compilation.second_module,
        compilation.receipt,
        limits,
      ) catch {
        _ => raise Invalid(FinalBinding)
      }
    }
  }
}

///|
/// Produces the complete BF16 release input from typed lowering candidates and
/// exact offline compiler outputs only. Actual CUBIN bytes establish catalog
/// families before full-graph contracts and strict family binding. No file,
/// process, compiler, CUDA, PTX, or JIT authority exists in this package.
pub fn produce_luna_bf16_kernel_release(
  model_plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  compilations : ArrayView[LunaBf16OfflineCompilation],
  limits : LunaBf16KernelProducerLimits,
) -> LunaBf16KernelReleaseInput raise LunaBf16KernelProducerError {
  let operations = model_plan.operations()
  if operations.length() > limits.max_operations ||
    compilations.length() > limits.max_operations {
    raise LimitExceeded(OperationCoverage, maximum=limits.max_operations)
  }
  if operations.length() != compilations.length() || operations.length() < 9 {
    raise Invalid(OperationCoverage)
  }
  if !layout.is_canonical_for(
      model_plan.identity(),
      model_plan.kv_cache_geometry(),
    ) ||
    !target.supports_bf16() {
    raise Mismatch(DeviceTarget)
  }
  require_complete_kinds(operations)

  let families : Array[ReusableFamilyBinding] = []
  let entries : Array[@catalog.CatalogEntry] = []
  let prepared : Array[PreparedCompilation] = []
  let launch_inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  let mut selected_profile : @launch_contract.PagedKernelProfile? = None
  let mut embedded_greedy_sampling = false
  for operation in operations {
    guard compilation_index(compilations, operation.id()) is Some(index) else {
      raise Invalid(OperationCoverage)
    }
    let compilation = compilations[index]
    let candidate = compilation.candidate
    if !candidate_kind_matches(candidate, operation.kind()) {
      raise Mismatch(OperationKind)
    }
    if candidate_target(candidate) != target {
      raise Mismatch(DeviceTarget)
    }
    match candidate {
      Projection(value) if value.family() == LanguageModelHead =>
        embedded_greedy_sampling = true
      _ => ()
    }
    let profile = candidate_profile(candidate)
    match selected_profile {
      None => selected_profile = Some(profile)
      Some(expected) => if expected != profile { raise Mismatch(Profile) }
    }
    if compilation.first_module.length() == 0 ||
      compilation.second_module.length() == 0 {
      raise Invalid(CompilerOutput)
    }
    let artifact_digest = @catalog.AotArtifactDigest::from_sha256(
      sha256_bytes(compilation.first_module),
    ) catch {
      _ => raise Invalid(CompilerOutput)
    }
    let semantics = operation_semantics(operation, layout)
    let operands = candidate_operands(candidate)
    let workspace_bytes = candidate_workspace_bytes(operation, profile)
    let workspace_alignment = candidate_workspace_alignment(
      operation, profile, operands,
    )
    let family = match
      reusable_binding_index(
        families, operation, profile, semantics, workspace_alignment,
      ) {
      Some(binding_index) => {
        let binding = families[binding_index]
        if binding.family.module_digest() != artifact_digest ||
          binding.entry_point_id != candidate_entry_point_id(candidate) ||
          binding.function_symbol != candidate_function_symbol(candidate) ||
          binding.source_digest_sha256 != candidate_source_digest(candidate) {
          raise Mismatch(ReusableFamily)
        }
        binding.family
      }
      None => {
        let family_id = @catalog.AotKernelFamilyId::from_stable_int(
          families.length() + 1,
        ) catch {
          _ => raise Invalid(Catalog)
        }
        let fresh = @catalog.AotKernelFamily::new(
          module_digest=artifact_digest,
          family_id~,
        )
        families.push({
          operation_kind: operation.kind(),
          operation_shape: operation.shape(),
          capability: operation.required_capability(),
          semantics,
          workspace_bytes,
          workspace_alignment,
          family: fresh,
          entry_point_id: candidate_entry_point_id(candidate),
          function_symbol: candidate_function_symbol(candidate),
          source_digest_sha256: candidate_source_digest(candidate),
        })
        entries.push(
          @catalog.CatalogEntry::new_paged_v3(
            operation_kind=operation.kind(),
            operation_shape=operation.shape(),
            capability=operation.required_capability(),
            execution_semantics=semantics,
            compute_major=target.compute_major(),
            compute_minor=target.compute_minor(),
            implementation=ContentAddressedAot(fresh),
            workspace_bytes~,
            workspace_alignment~,
          ) catch {
            _ => raise Invalid(Catalog)
          },
        )
        fresh
      }
    }
    let entry_point = @catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=candidate_entry_point_id(candidate),
    )
    launch_inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point~,
        dimensions=candidate_dimensions(candidate),
        operands~,
      ) catch {
        _ => raise Invalid(LaunchContracts)
      },
    )
    prepared.push({ compilation, operation })
  }
  guard selected_profile is Some(profile) else { raise Invalid(Profile) }
  let catalog = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v3(),
    entries~,
  ).resolve_paged(model_plan, layout, target) catch {
    _ => raise Invalid(Catalog)
  }
  let contracts = @launch_contract.admit_paged_graph(
    model_plan,
    catalog,
    layout,
    [profile],
    launch_inputs,
    limits.launch_contracts,
  ) catch {
    _ => raise Invalid(LaunchContracts)
  }
  let compiled : Array[@luna_kernel_bundle.LunaKernelCompiledOperation] = []
  for item in prepared {
    compiled.push(bind_compilation(item, contracts, limits.bundle))
  }
  let bundle = @luna_kernel_bundle.produce_luna_kernel_bundle(
    model_plan,
    target,
    contracts,
    compiled,
    limits.bundle,
  ) catch {
    _ => raise Invalid(Bundle)
  }
  { catalog, contracts, bundle, embedded_greedy_sampling }
}

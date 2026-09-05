///|
fn exact_attention_contract(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  operation : @plan.OperationId,
) -> @launch_contract.PagedKvAotLaunchContract raise DeviceWorkerError {
  let mut found : @launch_contract.PagedKvAotLaunchContract? = None
  for contract in contracts.contracts() {
    if contract.operation_id() == operation {
      if found is Some(_) || contract.operation_kind() != CausalAttention {
        raise Invalid(ResourceState)
      }
      found = Some(contract)
    }
  }
  guard found is Some(contract) else { raise Invalid(ResourceState) }
  contract
}

///|
/// Materializes one reusable split-K launch pair per Qwen attention layer at
/// startup. All bindings share immutable module bytes; the executor loads that
/// CUBIN once and resolves the two stable symbols once.
fn admit_reusable_qwen_decode_split_bindings(
  model_plan : @plan.ModelPlan,
  execution : @execution_manifest_file.PagedExecutionAdmission,
  bundle : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle?,
) -> ReadOnlyArray[
  @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
] raise DeviceWorkerError {
  guard bundle is Some(bundle) &&
    bundle.qwen_paged_attention_decode_split_module() is Some(module_) else {
    return ReadOnlyArray::from_array([])
  }
  let partial_entry = @catalog.AotKernelEntryPointId::from_stable_int(3903) catch {
    _ => raise Invalid(ResourceState)
  }
  let merge_entry = @catalog.AotKernelEntryPointId::from_stable_int(3904) catch {
    _ => raise Invalid(ResourceState)
  }
  let bindings : Array[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ] = []
  for operation in model_plan.operations() {
    if operation.kind() != CausalAttention {
      continue
    }
    let contract = exact_attention_contract(
      execution.contracts(),
      operation.id(),
    )
    bindings.push(
      @luna_cuda_paged_attention_aot.bind_reusable_paged_attention_decode_split_cuda_aot(
        operation,
        contract.profile(),
        execution.layout(),
        execution.blueprint().target(),
        contract.operands(),
        module_.module_bytes(),
        module_.module_digest().as_hex(),
        partial_entry,
        merge_entry,
        8,
        268435456L,
        compiler_readonly=module_.kernel_abi() ==
          PagedAttentionReadOnlyCompilerDecodeProductionV7,
      ) catch {
        _ => raise Invalid(ResourceState)
      },
    )
  }
  if bindings.is_empty() || bindings.length() > 64 {
    raise Invalid(ResourceState)
  }
  ReadOnlyArray::from_array(bindings)
}

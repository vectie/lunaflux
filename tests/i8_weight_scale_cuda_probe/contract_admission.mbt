///|
let probe_function_symbols : FixedArray[String] = [
  "lunaflux_probe_i8_embedding_v1", "lunaflux_probe_i8_rms_1_v1", "lunaflux_probe_i8_qkv_v1",
  "lunaflux_probe_i8_rope_v1", "lunaflux_probe_i8_paged_attention_v1", "lunaflux_probe_i8_weight_scale_projection_v1",
  "lunaflux_probe_i8_residual_1_v1", "lunaflux_probe_i8_rms_2_v1", "lunaflux_probe_i8_mlp_v1",
  "lunaflux_probe_i8_residual_2_v1", "lunaflux_probe_i8_rms_3_v1", "lunaflux_probe_i8_lm_head_v1",
]

///|
priv struct I8ProbeAdmission {
  device_numeric : @device.DeviceNumericCapability
  production_policy : @i8_admission.I8ProductionPolicyAdmission
  contracts : @launch_contract.PagedV4LaunchContractSet
  artifacts : @artifact.KernelArtifactBundle
  selected : @launch_contract.PagedV4AotLaunchContract
}

///|
fn require_selected_contract(
  plan : @plan.ModelPlan,
  contract : @launch_contract.PagedV4AotLaunchContract,
) -> Unit raise I8PhysicalProbeFailure {
  require_probe(contract.operation_id().as_int() == 5, "wrong I8 operation")
  require_probe(
    contract.operation_kind() == OutputProjection,
    "wrong I8 operation kind",
  )
  require_probe(
    contract.operation_execution_digest() ==
    @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1().digest(),
    "I8 execution digest is not production symmetric-I8 v1",
  )
  let operands = contract.operands()
  require_probe(operands.length() == 5, "I8 ABI operand count changed")
  guard operands[0].role() is RuntimeInput(StepCounts) &&
    operands[1].role() is ActivationInput(_) &&
    operands[2].role() is WeightInput(weight) &&
    operands[3].role() is WeightScaleInput(scale_weight, scale) &&
    operands[4].role() is ActivationOutput(_) else {
    raise I8PhysicalProbeFailure("I8 weight+scale ABI order changed")
  }
  require_probe(
    weight == scale_weight,
    "I8 scale is associated with another weight",
  )
  guard plan.tensor(weight) is Some(tensor) else {
    raise I8PhysicalProbeFailure("I8 ABI weight is absent from the plan")
  }
  require_probe(
    tensor.scale() == Some(scale),
    "I8 ABI scale is not schema-owned",
  )
  require_probe(operands[2].byte_count() == 16L, "I8 weight is not 4x4 bytes")
  require_probe(
    operands[3].byte_count() == 16L,
    "I8 scale is not four F32 values",
  )
}

///|
fn admit_i8_probe(
  module_bytes : Bytes,
  capability : @device.DeviceCapability,
) -> I8ProbeAdmission raise {
  let device_numeric = @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    capability,
  )
  let target = @catalog.DeviceTarget::from_capability(capability)
  let plan = probe_model_plan()
  let layout = probe_kv_layout(plan)
  let module_digest = probe_module_digest(module_bytes)
  let resolved = probe_resolved_catalog(plan, layout, target, module_digest)
  let production_policy = @i8_admission.admit_production_llama_symmetric_i8_weight_only_v1(
    plan,
    device_numeric,
    resolved,
    @i8_admission.I8InertKernelCapabilityAdmissionLimits::new(
      max_operations=16,
      max_weights=32,
      max_canonical_bytes=65536,
    ),
  )
  let contracts = probe_launch_contracts(plan, resolved, layout, module_digest)
  guard contracts.contract(probe_profile().id(), @plan.OperationId::new(5))
    is Some(selected) else {
    raise I8PhysicalProbeFailure("I8 launch contract is absent")
  }
  require_selected_contract(plan, selected)
  let entry_points : Array[@artifact.KernelEntryPointInput] = []
  let identities : Array[@catalog.AotKernelEntryPoint] = []
  for index, _operation in plan.operations() {
    let representative = probe_representative_operation(plan, index)
    let identity = probe_entry_point(module_digest, representative)
    if !identities.contains(identity) {
      identities.push(identity)
      entry_points.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=identity,
          function_symbol=probe_function_symbols[representative],
        ),
      )
    }
  }
  let artifacts = @artifact.admit_paged_v4(
    contracts,
    [
      @artifact.KernelModuleInput::from_bytes(
        digest=module_digest,
        module_bytes~,
      ),
    ],
    entry_points,
    @artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=16,
      max_module_bytes=1048576L,
      max_total_module_bytes=1048576L,
      max_function_symbol_bytes=64,
    ),
  )
  guard artifacts.entry_point(selected.entry_point()) is Some(entry) else {
    raise I8PhysicalProbeFailure("I8 artifact entry point is absent")
  }
  require_probe(
    entry.function_symbol() == probe_function_symbols[5],
    "I8 artifact symbol changed",
  )
  { device_numeric, production_policy, contracts, artifacts, selected }
}

///|
test "probe admits the exact production I8 catalog-v4 weight+scale contract" {
  let capability = @device.DeviceCapability::from_observed_facts(
    compute_major=8,
    compute_minor=9,
    total_memory_bytes=1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let admission = admit_i8_probe(b"static-probe-module", capability)
  assert_eq(
    admission.contracts.catalog_version(),
    @catalog.CatalogVersion::v4(),
  )
  assert_eq(admission.selected.operation_id().as_int(), 5)
  assert_eq(admission.selected.operands().length(), 5)
  assert_eq(admission.artifacts.module_count(), 1)
  assert_eq(
    admission.device_numeric.feature(),
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1,
  )
  assert_eq(admission.production_policy.i8_parameter_count(), 8)
  assert_eq(admission.production_policy.scale_tensor_count(), 8)
}

///|
test "probe admits sm120 as I8 software target without a physical claim" {
  let capability = @device.DeviceCapability::from_observed_facts(
    compute_major=12,
    compute_minor=0,
    total_memory_bytes=1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let admission = admit_i8_probe(b"static-sm120-probe-module", capability)
  assert_eq(admission.device_numeric.compute_major(), 12)
  assert_eq(admission.device_numeric.compute_minor(), 0)
  assert_eq(admission.production_policy.i8_parameter_count(), 8)
  assert_eq(admission.artifacts.module_count(), 1)
}

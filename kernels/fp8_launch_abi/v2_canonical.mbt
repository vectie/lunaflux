///|
fn v2_policy_tag(
  policy : @numeric_contract.ActivationScalePolicy,
) -> String raise Fp8LaunchAbiError {
  if policy ==
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() {
    "external-dynamic-per-tensor-f32-v1"
  } else if policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() {
    "gated-external-and-post-silu-dynamic-per-tensor-f32-v1"
  } else if policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() {
    "gated-external-and-post-silu-bound-expf-f32-v2"
  } else {
    raise InvalidFp8LaunchAbi(ActivationScalePolicy)
  }
}

///|
fn v2_stage_tag(
  stage : @numeric_contract.ActivationScaleStage,
) -> String raise Fp8LaunchAbiError {
  if stage ==
    @numeric_contract.ActivationScaleStage::external_operation_input_v1() {
    "external-operation-input-v1"
  } else if stage ==
    @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1() {
    "post-silu-gate-up-product-v1"
  } else {
    raise InvalidFp8LaunchAbi(ActivationScaleStage)
  }
}

///|
fn v2_operation_kind_tag(kind : @plan.OperationKind) -> String {
  match kind {
    QkvProjection => "qkv-projection"
    OutputProjection => "output-projection"
    GatedMlp => "gated-mlp"
    LanguageModelHead => "language-model-head"
    Embedding => "embedding"
    RmsNorm => "rms-norm"
    RotaryEmbedding => "rotary-embedding"
    CausalAttention => "causal-attention"
    ResidualAdd => "residual-add"
  }
}

///|
fn append_v2_operand_role(
  output : Array[Byte],
  role : @launch_contract.AotOperandRole,
) -> Unit {
  match role {
    ExternalInput(TokenIds) => append_field(output, "external-token-ids")
    WeightInput(weight) => {
      append_field(output, "weight")
      append_field(output, weight.as_int().to_string())
    }
    WeightScaleInput(weight, scale) => {
      append_field(output, "weight-scale")
      append_field(output, weight.as_int().to_string())
      append_field(output, scale.tensor().as_int().to_string())
    }
    ActivationInput(activation) => {
      append_field(output, "activation-input")
      append_field(output, activation.as_int().to_string())
    }
    RuntimeInput(runtime) => {
      append_field(output, "runtime-input")
      append_field(
        output,
        match runtime {
          StepCounts => "step-counts"
          QueryPositions => "query-positions"
          QueryRowOffsets => "query-row-offsets"
          SequenceLengths => "sequence-lengths"
          PageTableOffsets => "page-table-offsets"
          PageTablePageIndices => "page-table-page-indices"
        },
      )
    }
    PersistentKvComponent(layer, component) => {
      append_field(output, "persistent-kv")
      append_field(output, layer.as_int().to_string())
      append_field(
        output,
        match component {
          Key => "key"
          Value => "value"
        },
      )
    }
    ActivationOutput(activation) => {
      append_field(output, "activation-output")
      append_field(output, activation.as_int().to_string())
    }
    Workspace => append_field(output, "workspace")
  }
}

///|
fn append_v2_launch_contract(
  output : Array[Byte],
  selected : SelectedV2LaunchContract,
) -> Unit {
  append_field(
    output,
    match selected.source_version {
      PagedNumericV4 => "paged-numeric-v4"
    },
  )
  let dimensions = selected.dimensions
  append_field(output, dimensions.grid_x().to_string())
  append_field(output, dimensions.grid_y().to_string())
  append_field(output, dimensions.grid_z().to_string())
  append_field(output, dimensions.block_x().to_string())
  append_field(output, dimensions.block_y().to_string())
  append_field(output, dimensions.block_z().to_string())
  append_field(output, dimensions.shared_memory_bytes().to_string())
  append_field(output, selected.operands.length().to_string())
  for ordinal, operand in selected.operands {
    append_field(output, ordinal.to_string())
    append_v2_operand_role(output, operand.role())
    append_field(output, operand.byte_count().to_string())
    append_field(output, operand.alignment().to_string())
  }
  append_field(output, selected.workspace_bytes.to_string())
  append_field(output, selected.workspace_alignment.to_string())
}

///|
fn encode_staged_abi_v2(
  plan : @plan.ModelPlan,
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  selected : SelectedV2LaunchContract,
  operation_kind : @plan.OperationKind,
  policy : @numeric_contract.ActivationScalePolicy,
  shape : Fp8ExactExecutionShape,
  scale_tensor_regions : ReadOnlyArray[Fp8ScaleTensorStorageRegion],
  workspace : Fp8LaunchStorageRegion,
  stages : ReadOnlyArray[Fp8ActivationScaleStageRegion],
) -> Fp8LaunchAbiDigest raise Fp8LaunchAbiError {
  let output : Array[Byte] = []
  append_field(output, "lunaflux.fp8.staged-dynamic-launch-abi.v2")
  append_field(output, plan.identity().content().as_hex())
  append_field(output, plan.identity().plan().as_hex())
  append_field(output, plan.numeric_binding().digest().as_hex())
  append_field(output, manifest.digest().as_hex())
  append_field(output, manifest.target().compute_major().to_string())
  append_field(output, manifest.target().compute_minor().to_string())
  append_field(output, entry.profile().as_int().to_string())
  append_field(output, entry.operation().as_int().to_string())
  append_field(output, v2_operation_kind_tag(operation_kind))
  append_field(output, entry.execution_digest().as_hex())
  append_field(output, entry.module_digest().sha256())
  append_field(
    output,
    entry.entry_point().family().family_id().as_int().to_string(),
  )
  append_field(
    output,
    entry.entry_point().entry_point_id().as_int().to_string(),
  )
  append_v2_launch_contract(output, selected)
  append_shape(output, shape)
  append_field(output, v2_policy_tag(policy))
  append_field(output, entry.weights().length().to_string())
  for index, weight in entry.weights() {
    append_field(output, weight.tensor().as_int().to_string())
    append_field(output, weight.storage_digest().as_hex())
    append_field(output, weight.scale().tensor().as_int().to_string())
    append_field(output, weight.scale_storage_digest().as_hex())
    append_region(output, scale_tensor_regions[index].region)
  }
  append_region(output, workspace)
  append_field(output, stages.length().to_string())
  for stage in stages {
    append_field(output, stage.ordinal.to_string())
    append_field(output, v2_stage_tag(stage.stage))
    append_region(output, stage.region)
  }
  Fp8LaunchAbiDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(Bytes::from_array(output))),
  )
}

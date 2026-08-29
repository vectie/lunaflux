///|
fn require_v2_capacity(
  output : Array[Byte],
  additional : Int,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  if additional < 0 || output.length() > maximum - additional {
    raise InvalidFp8RuntimeRecipeV2(CanonicalBytes)
  }
}

///|
fn append_v2_text(
  output : Array[Byte],
  value : String,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  let bytes = @utf8.encode(value)
  require_v2_capacity(output, 8, maximum)
  for byte in bytes.length().to_int64().reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
  require_v2_capacity(output, bytes.length(), maximum)
  for byte in bytes {
    output.push(byte)
  }
}

///|
fn append_v2_region(
  output : Array[Byte],
  region : @fp8_launch_abi.Fp8LaunchStorageRegion,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  append_v2_text(output, region.storage().as_sha256(), maximum)
  append_v2_text(output, region.storage_bytes().to_string(), maximum)
  append_v2_text(output, region.offset().to_string(), maximum)
  append_v2_text(output, region.byte_length().to_string(), maximum)
  append_v2_text(output, region.alignment().to_string(), maximum)
}

///|
fn append_v2_stage(
  output : Array[Byte],
  stage : @fp8_launch_abi.Fp8ActivationScaleStageRegion,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  append_v2_text(output, stage.ordinal().to_string(), maximum)
  let tag = if stage.stage() ==
    @numeric_contract.ActivationScaleStage::external_operation_input_v1() {
    "external-operation-input-v1"
  } else if stage.stage() ==
    @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1() {
    "post-silu-gate-up-product-v1"
  } else {
    raise InvalidFp8RuntimeRecipeV2(StageBinding)
  }
  append_v2_text(output, tag, maximum)
  append_v2_region(output, stage.region(), maximum)
}

///|
fn append_v2_raw_operand_role(
  output : Array[Byte],
  role : @launch_contract.AotOperandRole,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  match role {
    ExternalInput(TokenIds) =>
      append_v2_text(output, "external-token-ids", maximum)
    WeightInput(weight) => {
      append_v2_text(output, "weight", maximum)
      append_v2_text(output, weight.as_int().to_string(), maximum)
    }
    WeightScaleInput(weight, scale) => {
      append_v2_text(output, "weight-scale", maximum)
      append_v2_text(output, weight.as_int().to_string(), maximum)
      append_v2_text(output, scale.tensor().as_int().to_string(), maximum)
    }
    ActivationInput(activation) => {
      append_v2_text(output, "activation-input", maximum)
      append_v2_text(output, activation.as_int().to_string(), maximum)
    }
    ActivationOutput(activation) => {
      append_v2_text(output, "activation-output", maximum)
      append_v2_text(output, activation.as_int().to_string(), maximum)
    }
    RuntimeInput(runtime) => {
      append_v2_text(output, "runtime-input", maximum)
      append_v2_text(
        output,
        match runtime {
          StepCounts => "step-counts"
          QueryPositions => "query-positions"
          QueryRowOffsets => "query-row-offsets"
          SequenceLengths => "sequence-lengths"
          PageTableOffsets => "page-table-offsets"
          PageTablePageIndices => "page-table-page-indices"
        },
        maximum,
      )
    }
    PersistentKvComponent(layer, component) => {
      append_v2_text(output, "persistent-kv", maximum)
      append_v2_text(output, layer.as_int().to_string(), maximum)
      append_v2_text(
        output,
        match component {
          Key => "key"
          Value => "value"
        },
        maximum,
      )
    }
    Workspace => append_v2_text(output, "workspace", maximum)
  }
}

///|
fn append_v2_raw_launch_authority(
  output : Array[Byte],
  launch : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  append_v2_text(output, "paged-numeric-v4", maximum)
  let dimensions = launch.launch_dimensions()
  append_v2_text(output, dimensions.grid_x().to_string(), maximum)
  append_v2_text(output, dimensions.grid_y().to_string(), maximum)
  append_v2_text(output, dimensions.grid_z().to_string(), maximum)
  append_v2_text(output, dimensions.block_x().to_string(), maximum)
  append_v2_text(output, dimensions.block_y().to_string(), maximum)
  append_v2_text(output, dimensions.block_z().to_string(), maximum)
  append_v2_text(output, dimensions.shared_memory_bytes().to_string(), maximum)
  let operands = launch.raw_operands()
  append_v2_text(output, operands.length().to_string(), maximum)
  for ordinal, operand in operands {
    append_v2_text(output, ordinal.to_string(), maximum)
    append_v2_raw_operand_role(output, operand.role(), maximum)
    append_v2_text(output, operand.byte_count().to_string(), maximum)
    append_v2_text(output, operand.alignment().to_string(), maximum)
  }
}

///|
fn encode_runtime_recipe_v2(
  startup : @fp8_startup_admission.Fp8StartupAdmissionV2,
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launches : ReadOnlyArray[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi],
  bindings : ReadOnlyArray[Fp8LaunchWeightBindingV2],
  workspaces : ReadOnlyArray[Fp8WorkspaceBindingV2],
  weight_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  workspace_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  workspace_capacity_bytes : Int64,
  maximum : Int,
) -> Bytes raise Fp8RuntimeRecipeV2Error {
  let output : Array[Byte] = []
  append_v2_text(output, "lunaflux.fp8-runtime-recipe.v2", maximum)
  append_v2_text(output, startup.digest().as_hex(), maximum)
  append_v2_text(output, weights.artifact_digest().as_hex(), maximum)
  append_v2_text(output, weight_storage.as_sha256(), maximum)
  append_v2_text(output, workspace_storage.as_sha256(), maximum)
  append_v2_text(output, workspace_capacity_bytes.to_string(), maximum)
  append_v2_text(output, weights.layout().arena_bytes().to_string(), maximum)
  append_v2_text(
    output,
    weights.layout().alignment_bytes().to_string(),
    maximum,
  )
  append_v2_text(output, weights.layout().length().to_string(), maximum)
  for index in 0..<weights.layout().length() {
    let reference = try! @plan.TensorRef::new(index)
    guard weights.layout().region(reference) is Some(region) else {
      raise InvalidFp8RuntimeRecipeV2(WeightLayout)
    }
    append_v2_text(output, reference.as_int().to_string(), maximum)
    append_v2_text(output, region.offset_bytes().to_string(), maximum)
    append_v2_text(output, region.byte_length().to_string(), maximum)
  }
  append_v2_text(output, launches.length().to_string(), maximum)
  for index, launch in launches {
    append_v2_text(output, launch.digest().as_sha256(), maximum)
    append_v2_raw_launch_authority(output, launch, maximum)
    let binding = bindings[index]
    append_v2_text(output, binding.profile().as_int().to_string(), maximum)
    append_v2_text(output, binding.operation().as_int().to_string(), maximum)
    for region in binding.regions() {
      append_v2_text(output, region.tensor().as_int().to_string(), maximum)
      append_v2_text(output, region.storage_digest().as_hex(), maximum)
      append_v2_region(output, region.region(), maximum)
    }
    let scales = launch.scale_tensor_regions()
    append_v2_text(output, scales.length().to_string(), maximum)
    for scale in scales {
      append_v2_text(
        output,
        scale.tensor().tensor().as_int().to_string(),
        maximum,
      )
      append_v2_text(output, scale.storage_digest().as_hex(), maximum)
      append_v2_region(output, scale.region(), maximum)
    }
    append_v2_region(output, workspaces[index].workspace(), maximum)
    append_v2_text(output, launch.stages().length().to_string(), maximum)
    for stage in launch.stages() {
      append_v2_stage(output, stage, maximum)
    }
  }
  Bytes::from_array(output)
}

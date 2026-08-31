///|
fn family_tag(family : PointwiseAotFamily) -> String {
  match family {
    EmbeddingLookup => "embedding_lookup"
    RmsNorm => "rms_norm"
    QkRmsNorm => "qk_rms_norm"
    PositionedRotary => "positioned_rotary"
    ResidualAdd => "residual_add"
    KvCacheWrite => "kv_cache_write"
  }
}

///|
fn numerical_tag(contract : PointwiseNumericalContract) -> String {
  match contract {
    BitExactBf16Copy => "bit_exact_bf16_copy"
    OrderedBf16Add => "ordered_bf16_add"
    Float32ReductionToleranceV1 => "float32_reduction_atol_2e-3_rtol_2e-3"
    Float32TranscendentalToleranceV1 =>
      "float32_transcendental_atol_2e-3_rtol_2e-3"
  }
}

///|
fn bool_tag(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
pub fn canonical_cuda_float_literal(value : Double) -> String {
  let text = value.to_string()
  if text.contains(".") || text.contains("e") || text.contains("E") {
    text + "f"
  } else {
    text + ".0f"
  }
}

///|
fn operand_role_tag(role : @launch_contract.AotOperandRole) -> String {
  match role {
    RuntimeInput(runtime) => {
      let tag = match runtime {
        StepCounts => "step_counts"
        QueryPositions => "query_positions"
        QueryRowOffsets => "query_row_offsets"
        SequenceLengths => "sequence_lengths"
        PageTableOffsets => "page_table_offsets"
        PageTablePageIndices => "page_table_page_indices"
      }
      "runtime:" + tag
    }
    ExternalInput(TokenIds) => "external:token_ids"
    WeightInput(reference) => "weight:" + reference.as_int().to_string()
    ActivationInput(reference) =>
      "activation_in:" + reference.as_int().to_string()
    ActivationOutput(reference) =>
      "activation_out:" + reference.as_int().to_string()
    PersistentKvComponent(layer, component) => {
      let component_tag = match component {
        Key => "key"
        Value => "value"
      }
      "kv:" + layer.as_int().to_string() + ":" + component_tag
    }
    Workspace => "workspace"
    WeightScaleInput(reference, scale) =>
      "weight_scale:" +
      reference.as_int().to_string() +
      ":" +
      scale.tensor().as_int().to_string()
  }
}

///|
fn operand_recipe(operands : ArrayView[@launch_contract.AotOperand]) -> String {
  let result = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      result.write_char(',')
    }
    result.write_string(operand_role_tag(operand.role()))
    result.write_char('@')
    result.write_string(operand.byte_count().to_string())
    result.write_char('/')
    result.write_string(operand.alignment().to_string())
  }
  result.to_string()
}

///|
fn pointwise_dimensions(
  family : PointwiseAotFamily,
  maximum_work_items : Int64,
) -> @launch_contract.AotLaunchDimensions raise PointwiseAotError {
  if maximum_work_items <= 0L {
    raise Invalid(LaunchDimensions)
  }
  let threads = if family == PositionedRotary { 32L } else { 256L }
  let reduction_family = family == RmsNorm || family == QkRmsNorm
  if !reduction_family && maximum_work_items > 9223372036854775552L {
    raise SizeOverflow(LaunchDimensions)
  }
  let grid_x = match family {
    RmsNorm => Int64::min(maximum_work_items, 64L)
    QkRmsNorm => Int64::min((maximum_work_items + 7L) / 8L, 32L)
    PositionedRotary => Int64::min(maximum_work_items, 64L)
    _ => Int64::min((maximum_work_items + threads - 1L) / threads, 32L)
  }
  if grid_x <= 0L || grid_x > 2147483647L {
    raise LimitExceeded(LaunchDimensions)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid_x.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=threads.to_int(),
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(LaunchDimensions)
  }
}

///|
fn pointwise_symbol(
  family : PointwiseAotFamily,
  operation : @plan.OperationId,
  entry_point_id : @catalog.AotKernelEntryPointId,
) -> String {
  "lunaflux_luna_" +
  family_tag(family) +
  "_v1_op_" +
  operation.as_int().to_string() +
  "_ep_" +
  entry_point_id.as_int().to_string()
}

///|
fn pointwise_contract(
  family : PointwiseAotFamily,
) -> PointwiseNumericalContract {
  match family {
    EmbeddingLookup | KvCacheWrite => BitExactBf16Copy
    ResidualAdd => OrderedBf16Add
    RmsNorm | QkRmsNorm => Float32ReductionToleranceV1
    PositionedRotary => Float32TranscendentalToleranceV1
  }
}

///|
fn pointwise_recipe(
  family : PointwiseAotFamily,
  operation : @plan.OperationId,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  numerical_contract : PointwiseNumericalContract,
  manifest_bindable : Bool,
  function_symbol : String,
  source_digest : PointwiseSourceDigest,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-pointwise-aot-recipe-v1\n" +
    "family=" +
    family_tag(family) +
    "\n" +
    "operation_id=" +
    operation.as_int().to_string() +
    "\n" +
    "entry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "compiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "output=cubin\noptimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    bool_tag(compiler.fused_multiply_add()) +
    "\n" +
    "reassociate=false\nmax_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    ",1,1\nblock=256,1,1\n" +
    "manifest_bindable=" +
    bool_tag(manifest_bindable) +
    "\n" +
    "numerical_contract=" +
    numerical_tag(numerical_contract) +
    "\n" +
    "operands=" +
    operand_recipe(operands) +
    "\n",
  )
}

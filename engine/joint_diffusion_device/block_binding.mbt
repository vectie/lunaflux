///|
/// Exact-shape admitted functions; RMSNorm/affine/residual are shared by the
/// attention and MLP halves. No function loading or compilation occurs here.
pub struct BlockFunctions {
  priv attention : AttentionFunctions
  priv norm : @device.Function
  priv affine : @device.Function
  priv residual : @device.Function
  priv gate_up : @device.Function
  priv down : @device.Function
}

///|
pub fn BlockFunctions::new(
  attention : AttentionFunctions,
  norm~ : @device.Function,
  affine~ : @device.Function,
  residual~ : @device.Function,
  gate_up~ : @device.Function,
  down~ : @device.Function,
) -> Self {
  { attention, norm, affine, residual, gate_up, down, }
}

///|
/// Assemble one complete joint block using this workspace. Residual state is
/// updated in place by elementwise gated residual kernels. The weight arena
/// contains separately supplied attention/MLP offsets plus two contiguous
/// hidden-width RMSNorm vectors. Modulation is six contiguous compact tables:
/// attention shift/scale/gate, then MLP shift/scale/gate. Parameters must already
/// have been produced for this layer and step. All external storage is borrowed
/// and must be disjoint from workspace; this routine queues no work.
pub fn BlockWorkspace::bind_block(
  self : Self,
  functions : BlockFunctions,
  rotary : RotaryBindings,
  modulation : ModulationBindings,
  state~ : @device.Allocation,
  state_offset~ : Int64,
  weights~ : @device.Allocation,
  attention_weight_offset~ : Int64,
  mlp_weight_offset~ : Int64,
  norm_weight_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let plan = self.layout
  guard rotary.rows == plan.rows &&
    rotary.width <= plan.dimension &&
    modulation.rows == plan.rows &&
    modulation.width == plan.hidden else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let table_bytes = modulation.parameter_rows.to_int64() *
    plan.hidden.to_int64() *
    2L
  // Validate complete external spans before adding per-half offsets.
  ignore(
    @device.KernelArgument::device_region(
      parameters,
      offset=parameter_offset,
      byte_count=table_bytes * 6L,
      alignment=2L,
    ),
  )
  ignore(
    @device.KernelArgument::device_region(
      weights,
      offset=norm_weight_offset,
      byte_count=plan.hidden.to_int64() * 4L,
      alignment=2L,
    ),
  )
  fn prefix(
    half : Int,
  ) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
    let norm = block_rms_norm_launch(
      functions.norm,
      rows=plan.rows,
      width=plan.hidden,
      input=state,
      input_offset=state_offset,
      weights~,
      weight_offset=norm_weight_offset +
        half.to_int64() * plan.hidden.to_int64() * 2L,
      output=self.allocation,
      output_offset=plan.normalized_offset,
    )
    let affine = modulation.launch(
      functions.affine,
      input=self.allocation,
      input_offset=plan.normalized_offset,
      parameter=parameters,
      parameter_offset=parameter_offset + half.to_int64() * 3L * table_bytes,
      third=parameters,
      third_offset=parameter_offset + (half.to_int64() * 3L + 1L) * table_bytes,
      output=self.allocation,
      output_offset=plan.normalized_offset,
      residual=false,
    )
    [norm, affine]
  }
  fn residual(
    half : Int,
  ) -> @device.OrderedKernelLaunch raise @device.DeviceError {
    modulation.launch(
      functions.residual,
      input=state,
      input_offset=state_offset,
      parameter=parameters,
      parameter_offset=parameter_offset +
        (half.to_int64() * 3L + 2L) * table_bytes,
      third=self.allocation,
      third_offset=plan.branch_offset,
      output=state,
      output_offset=state_offset,
      residual=true,
    )
  }
  let attention = rotary.attention_launches(
    functions.attention,
    input=self.allocation,
    input_offset=plan.normalized_offset,
    output=self.allocation,
    output_offset=plan.branch_offset,
    scratch=self.allocation,
    scratch_offset=plan.scratch_offset,
    weights~,
    weight_offset=attention_weight_offset,
    hidden_width=plan.hidden,
    head_count=plan.heads,
    head_dimension=plan.dimension,
  )
  let mlp = StagedMlpLayout::new(
    rows=plan.rows,
    hidden_width=plan.hidden,
    intermediate_width=plan.intermediate,
  ).launches(
    functions.gate_up,
    functions.down,
    input=self.allocation,
    input_offset=plan.normalized_offset,
    output=self.allocation,
    output_offset=plan.branch_offset,
    scratch=self.allocation,
    scratch_offset=plan.scratch_offset,
    weights~,
    weight_offset=mlp_weight_offset,
  )
  transformer_block_launches(
    prefix(0),
    attention,
    residual(0),
    prefix(1),
    mlp,
    residual(1),
  )
}

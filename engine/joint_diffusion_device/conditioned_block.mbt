///|
/// Bind parameter production and all block consumers as one ordered unit.
/// Conditioning is pre-activated BF16, one row per distinct timestep. Each
/// timestep generates modality_count parameter rows with six fields per row.
/// Parameters/workspace may be shared by sequential blocks, never concurrent
/// queues. Model-family resolution and function/weight loading occur upstream.
pub fn BlockWorkspace::bind_conditioned_block(
  self : Self,
  functions : BlockFunctions,
  parameter_function : @device.Function,
  rotary : RotaryBindings,
  modulation : ModulationBindings,
  conditioning~ : @device.Allocation,
  conditioning_offset~ : Int64,
  conditioning_width~ : Int,
  state~ : @device.Allocation,
  state_offset~ : Int64,
  weights~ : @device.Allocation,
  attention_weight_offset~ : Int64,
  mlp_weight_offset~ : Int64,
  norm_weight_offset~ : Int64,
  parameter_weight_offset~ : Int64,
  parameter_bias_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
  modality_count~ : Int,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let rows = conditioning_rows(modulation.parameter_rows, modality_count)
  let producer = ParameterProjectionLayout::new(
    rows~,
    input_width=conditioning_width,
    width=self.layout.hidden,
    groups=modality_count,
    fields=6,
  ).launch(
    parameter_function,
    input=conditioning,
    input_offset=conditioning_offset,
    weights~,
    weight_offset=parameter_weight_offset,
    bias_offset=parameter_bias_offset,
    output=parameters,
    output_offset=parameter_offset,
  )
  let consumers = self.bind_block(
    functions,
    rotary,
    modulation,
    state~,
    state_offset~,
    weights~,
    attention_weight_offset~,
    mlp_weight_offset~,
    norm_weight_offset~,
    parameters~,
    parameter_offset~,
  )
  let launches = [producer]
  launches.append(consumers)
  launches
}

///|
fn conditioning_rows(
  parameter_rows : Int,
  modalities : Int,
) -> Int raise @device.DeviceError {
  guard modalities > 0 && parameter_rows > 0 && parameter_rows % modalities == 0 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  parameter_rows / modalities
}

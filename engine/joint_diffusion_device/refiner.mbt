///|
/// Standard pre-norm block, without modulation or rotary. Functions must be
/// compiled for this workspace's live text rows, not padded multimodal rows.
pub fn BlockWorkspace::bind_refiner(
  self : Self,
  attention : AttentionFunctions,
  norm~ : @device.Function,
  residual~ : @device.Function,
  gate_up~ : @device.Function,
  down~ : @device.Function,
  state~ : @device.Allocation,
  state_offset~ : Int64,
  weights~ : @device.Allocation,
  attention_weight_offset~ : Int64,
  mlp_weight_offset~ : Int64,
  norm_weight_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let p = self.layout
  let a = self.allocation
  fn normalization(
    half : Int,
  ) -> @device.OrderedKernelLaunch raise @device.DeviceError {
    block_rms_norm_launch(
      norm,
      rows=p.rows,
      width=p.hidden,
      input=state,
      input_offset=state_offset,
      weights~,
      weight_offset=norm_weight_offset +
        half.to_int64() * p.hidden.to_int64() * 2L,
      output=a,
      output_offset=p.normalized_offset,
    )
  }
  fn addition() -> @device.OrderedKernelLaunch raise @device.DeviceError {
    let target = @device.KernelArgument::device_region(
      state,
      offset=state_offset,
      byte_count=p.hidden_bytes,
      alignment=2L,
    )
    @device.OrderedKernelLaunch::new(
      residual,
      linear_dimensions(p.hidden_bytes / 2L, 256),
      @device.KernelArguments::new([
        target,
        @device.KernelArgument::device_region(
          a,
          offset=p.branch_offset,
          byte_count=p.hidden_bytes,
          alignment=2L,
        ),
        target,
      ]),
    )
  }
  let att = attention.launches(
    rows=p.rows,
    input=a,
    input_offset=p.normalized_offset,
    output=a,
    output_offset=p.branch_offset,
    scratch=a,
    scratch_offset=p.scratch_offset,
    weights~,
    weight_offset=attention_weight_offset,
    hidden_width=p.hidden,
    head_count=p.heads,
    head_dimension=p.dimension,
  )
  let mlp = StagedMlpLayout::new(
    rows=p.rows,
    hidden_width=p.hidden,
    intermediate_width=p.intermediate,
  ).launches(
    gate_up,
    down,
    input=a,
    input_offset=p.normalized_offset,
    output=a,
    output_offset=p.branch_offset,
    scratch=a,
    scratch_offset=p.scratch_offset,
    weights~,
    weight_offset=mlp_weight_offset,
  )
  compose_refiner(
    normalization(0),
    att,
    addition(),
    normalization(1),
    mlp,
    addition(),
  )
}

///|
fn[T] compose_refiner(
  norm1 : T,
  attention : ArrayView[T],
  residual1 : T,
  norm2 : T,
  mlp : ArrayView[T],
  residual2 : T,
) -> Array[T] raise @device.DeviceError {
  guard attention.length() == 4 && mlp.length() == 2 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let stages = [norm1]
  stages.append(attention)
  stages.push(residual1)
  stages.push(norm2)
  stages.append(mlp)
  stages.push(residual2)
  stages
}

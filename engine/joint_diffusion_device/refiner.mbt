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
  self
  .bind_refiner_stages(
    attention,
    norm~,
    residual~,
    gate_up~,
    down~,
    state~,
    state_offset~,
    weights~,
    attention_weight_offset~,
    mlp_weight_offset~,
    norm_weight_offset~,
  )
  .map(pair => pair.1)
}

///|
pub(all) enum RefinerStage {
  Norm1
  Qkv
  QkNorm
  Attention
  Projection
  AttentionResidual
  Norm2
  GateUp
  Down
  MlpResidual
}

///|
/// Immutable prepared order. Offline observers consume the same description
/// as ordinary binding, never reconstructing stage indices from kernel counts.
pub fn BlockWorkspace::bind_refiner_stages(
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
) -> Array[(RefinerStage, @device.OrderedKernelLaunch)] raise @device.DeviceError {
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
  describe_refiner(
    normalization(0),
    att,
    addition(),
    normalization(1),
    mlp,
    addition(),
  )
}

///|
fn[T] describe_refiner(
  norm1 : T,
  attention : ArrayView[T],
  residual1 : T,
  norm2 : T,
  mlp : ArrayView[T],
  residual2 : T,
) -> Array[(RefinerStage, T)] raise @device.DeviceError {
  guard attention.length() == 4 && mlp.length() == 2 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  [
    (Norm1, norm1),
    (Qkv, attention[0]),
    (QkNorm, attention[1]),
    (Attention, attention[2]),
    (Projection, attention[3]),
    (AttentionResidual, residual1),
    (Norm2, norm2),
    (GateUp, mlp[0]),
    (Down, mlp[1]),
    (MlpResidual, residual2),
  ]
}

///|
/// Diagnostic readback only, after the bound stream has been retired. State
/// residuals live externally and are read by their owner, not this workspace.
pub fn BlockWorkspace::capture_refiner_stage(
  self : Self,
  stage : RefinerStage,
) -> Bytes raise @device.DeviceError {
  let p = self.layout
  let (offset, count) = match stage {
    Norm1 | Norm2 => (p.normalized_offset, p.hidden_bytes)
    Qkv | QkNorm => (p.scratch_offset, p.hidden_bytes * 3L)
    Attention => (p.scratch_offset + p.hidden_bytes * 3L, p.hidden_bytes)
    Projection | Down => (p.branch_offset, p.hidden_bytes)
    GateUp =>
      (p.scratch_offset, p.rows.to_int64() * p.intermediate.to_int64() * 2L)
    AttentionResidual | MlpResidual =>
      raise @device.DeviceError::InvalidArgument(CopyToHost)
  }
  self.allocation.copy_to_host(offset, count)
}

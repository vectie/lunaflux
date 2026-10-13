///|
/// The model builder selects recurrent semantics; device scheduling is absent.
pub fn recurrent_compute_plan(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
  evaluation? : @precision_ir.RecurrentEvaluationLaw = OrderedTokenFloat32,
  tensor_group? : (Int, Int),
) -> @precision_ir.RecurrentDeltaPrecision raise {
  guard spec.attention_kind(layer_index) is KimiDeltaAttention else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  @precision_ir.RecurrentDeltaPrecision::new(
    head_count=partition_width(spec.linear_head_count(), tensor_group),
    key_dimension=spec.linear_head_dimension(),
    value_dimension=spec.linear_head_dimension(),
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=spec.linear_gate_lower_bound(),
    maximum_state_bytes~,
    evaluation~,
  )
}

///|
/// Q/K/V share a depthwise convolution geometry in KDA. This maps model
/// semantics to a device-neutral request-history plan, not to a CUDA schedule.
pub fn recurrent_convolution_plan(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
  tensor_group? : (Int, Int),
) -> @precision_ir.CausalConvolutionPrecision raise {
  guard spec.attention_kind(layer_index) is KimiDeltaAttention else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  @precision_ir.CausalConvolutionPrecision::new(
    channel_count=partition_width(spec.linear_head_count(), tensor_group) *
      spec.linear_head_dimension(),
    kernel_size=spec.linear_conv_kernel_size(),
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    maximum_state_bytes~,
  )
}

///|
/// Compose the complete KDA branch without exposing CUDA schedules to the model.
pub fn recurrent_block_plan(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
  evaluation? : @precision_ir.RecurrentEvaluationLaw = OrderedTokenFloat32,
  matrix_law? : @precision_ir.DenseDotLaw = ScalarFloat32,
  tensor_group? : (Int, Int),
) -> @precision_ir.RecurrentBlockPrecision raise {
  @precision_ir.RecurrentBlockPrecision::new(
    recurrent_compute_plan(
      spec,
      layer_index~,
      maximum_rows~,
      maximum_sequences~,
      cache_slots~,
      maximum_state_bytes~,
      evaluation~,
      tensor_group?,
    ),
    recurrent_convolution_plan(
      spec,
      layer_index~,
      maximum_rows~,
      maximum_sequences~,
      cache_slots~,
      maximum_state_bytes~,
      tensor_group?,
    ),
    hidden_width=spec.hidden_size(),
    control_rank=spec.linear_head_dimension(),
    output_normalization_epsilon=spec.rms_norm_epsilon(),
    projection_law=matrix_law,
    publication=if tensor_group is Some(_) {
      ReduceFloat32BeforeRound
    } else {
      LocalBFloat16
    },
  )
}

///|
fn partition_width(width : Int, group : (Int, Int)?) -> Int raise {
  match group {
    None => width
    Some((rank, world_size)) => {
      let (_, local) = @precision_ir.TensorAxisPartition::new(
        width,
        rank~,
        world_size~,
      ).range()
      local
    }
  }
}

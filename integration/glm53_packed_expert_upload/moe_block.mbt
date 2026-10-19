///|
/// Complete recurrent GLM block with automatic routing, packed NVFP4 routed
/// experts, BF16 shared expert and both residual-stream envelopes. All experts
/// for this layer are resident locally; partitioned execution requires the
/// existing collective-aware MoE program and is not silently substituted here.
pub fn GlmRecurrentLayer::with_moe_block(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int32,
  maximum_rows~ : Int32,
  maximum_sequences~ : Int32,
  cache_slots~ : Int32,
  maximum_state_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
  evaluation~ : @precision_ir.RecurrentEvaluationLaw = OrderedTokenFloat32,
  matrix_law~ : @precision_ir.DenseDotLaw = ScalarFloat32,
  matrix_staging~ : @matrix.MatrixStaging = Synchronous,
  hc_reduction~ : @precision_ir.HyperControlReduction = Ordered,
  expert_scaled_dot_law~ : @precision_ir.ScaledDotReductionLaw = DecodeOperandsFloat32,
  expert_transport~ : @matrix.ExpertTransport = IndexedRows,
  shared_matrix_columns~ : Int32 = 64,
  expert_fragment_slots~ : Int32 = 1,
  expert_scale_partitions~ : Int32 = 1,
  moe_reduction~ : @precision_ir.MoeReductionPlacement = ExpertPartialsFloat32,
  tensor_group? : (Int32, Int32),
) -> Self raise {
  guard spec.feed_forward_kind(layer_index) is RoutedMixtureOfExperts else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let branch = GlmRecurrentLayer::with_hyper_connection(
    spec,
    layer_index~,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    maximum_state_bytes~,
    evaluation~,
    matrix_law~,
    matrix_staging~,
    hc_reduction~,
    tensor_group?,
  )
  {
    spec,
    layer_index,
    program: branch.program,
    weights: branch.weights,
    hyper: branch.hyper,
    tensor_group,
    ffn: Some(
      make_moe_ffn(
        spec,
        layer_index,
        maximum_rows,
        maximum_expert_bytes,
        accuracy,
        hc_reduction~,
        matrix_law~,
        expert_scaled_dot_law~,
        expert_transport~,
        shared_matrix_columns~,
        expert_fragment_slots~,
        expert_scale_partitions~,
        moe_reduction~,
        tensor_group?,
      ),
    ),
    status: Fresh,
  }
}

///|
/// Shared model-level FFN construction for recurrent and sparse blocks. The
/// attention family does not alter routing, expert arithmetic or ownership.
fn make_moe_ffn(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index : Int32,
  maximum_rows : Int32,
  maximum_expert_bytes : Int64,
  accuracy : @precision_ir.Accuracy,
  hc_reduction~ : @precision_ir.HyperControlReduction = Ordered,
  matrix_law~ : @precision_ir.DenseDotLaw = ScalarFloat32,
  expert_scaled_dot_law~ : @precision_ir.ScaledDotReductionLaw = DecodeOperandsFloat32,
  expert_transport~ : @matrix.ExpertTransport = IndexedRows,
  shared_matrix_columns~ : Int32 = 64,
  expert_fragment_slots~ : Int32 = 1,
  expert_scale_partitions~ : Int32 = 1,
  moe_reduction~ : @precision_ir.MoeReductionPlacement = ExpertPartialsFloat32,
  tensor_group? : (Int32, Int32),
) -> GlmFfn raise {
  guard spec.feed_forward_kind(layer_index) is RoutedMixtureOfExperts else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let partition = match tensor_group {
    None => None
    Some((rank, world_size)) =>
      Some(
        tensor_expert_compute_plan(
          layer_index~,
          maximum_rows~,
          rank~,
          world_size~,
          accuracy~,
          matrix_law~,
          expert_scaled_dot_law~,
          memory_ceiling=maximum_expert_bytes,
        ),
      )
  }
  let routed = match partition {
    Some(partition) => partition.rank_plan()
    None =>
      expert_compute_plan(
        layer_index~,
        expert_indices=Array::makei(spec.routed_expert_count(), i => i),
        maximum_rows~,
        memory_ceiling=maximum_expert_bytes,
        accuracy~,
        matrix_law~,
        expert_scaled_dot_law~,
      )
  }
  let shared = shared_expert_compute_plan(
    layer_index~,
    maximum_rows~,
    memory_ceiling=maximum_expert_bytes,
    accuracy~,
    matrix_law~,
  )
  let shared_partition = match tensor_group {
    None => None
    Some((rank, world_size)) =>
      Some(
        @precision_ir.ExpertTensorPartition::new(
          shared,
          rank~,
          world_size~,
          memory_ceiling=maximum_expert_bytes,
        ),
      )
  }
  let shared = match shared_partition {
    None => shared
    Some(partition) => partition.rank_plan()
  }
  guard routed.bank_bytes() <= maximum_expert_bytes &&
    shared.bank_bytes() <= maximum_expert_bytes - routed.bank_bytes() else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  let program = @packed_expert_mlp.MoeProgram::new(
    // Two-rank F32 sums keep exactly the same pair of operands per lane;
    // broader topologies require their own reduction-order validation.
    reduction_placement=moe_reduction,
    coalesce_reductions=moe_reduction == ExpertPartialsFloat32 &&
      matrix_law == ShapeAwareFloat32 &&
      tensor_group is Some((_, 2)),
    @expert_source.packed_expert_mlp_source(
      routed,
      symbol="glm_routed_" + layer_index.to_string(),
      grouping?=if expert_transport == IndexedRows &&
        expert_fragment_slots == 1 &&
        expert_scale_partitions == 1 {
        None
      } else {
        Some(
          @matrix.ExpertGroupingPlan::new(
            maximum_rows~,
            selected_experts=routed.geometry().4,
            transport=expert_transport,
            shared_input_staging=expert_transport
              is (PackedBulk(..) | PackedBulkTiles(..)),
            fragment_slots=expert_fragment_slots,
            scale_partitions=expert_scale_partitions,
          ),
        )
      },
    ),
    @expert_source.packed_expert_mlp_source(
      shared,
      symbol="glm_shared_" + layer_index.to_string(),
      grouping?=if shared_matrix_columns == 64 {
        None
      } else {
        Some(
          @matrix.ExpertGroupingPlan::new(
            maximum_rows~,
            selected_experts=1,
            shared_input_staging=true,
            matrix_columns=shared_matrix_columns,
          ),
        )
      },
    ),
    @combine_source.moe_combine_source(
      @precision_ir.MoeCombinePrecision::new(
        maximum_rows~,
        hidden=spec.hidden_size(),
        rounding=Bf16RoutedThenSum,
        memory_ceiling=maximum_rows.to_int64() *
          spec.hidden_size().to_int64() *
          10L,
      ),
      symbol="glm_moe_combine_" + layer_index.to_string(),
    ),
    routing=@router_source.routing_source(
      routing_compute_plan(
        matrix_law~,
        maximum_rows~,
        maximum_workspace_bytes=maximum_expert_bytes,
        accuracy~,
      ),
      symbol="glm_router_" + layer_index.to_string(),
      parallel_selection=matrix_law == ShapeAwareFloat32,
    ),
  )
  let envelope = @precision_ir.HyperConnectionPrecision::new(
    stream_count=spec.hyper_connection_streams(),
    hidden_width=spec.hidden_size(),
    maximum_rows~,
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    mix_law=RoundedBf16ControlsAndProducts,
    control_square_law=hc_reduction.square_law(),
    control_dot_law=hc_reduction.dot_law(),
  )
  {
    envelope: @hyper.HyperConnectionFrame::new(
      @hyper_source.hyper_connection_source(
        envelope,
        symbol_prefix="glm_hyper_ffn_" + layer_index.to_string(),
      ),
    ),
    branch: Moe({
      routed,
      shared,
      program,
      source: program.source_bytes(),
      banks: [],
      tensor_partition: partition,
      shared_partition,
      tensor_boundary: None,
    }),
    input_bytes: maximum_rows.to_int64() *
    spec.hyper_connection_streams().to_int64() *
    spec.hidden_size().to_int64() *
    2L,
    input: None,
    launches: [],
  }
}

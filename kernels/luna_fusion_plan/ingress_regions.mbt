///|
/// Ingress is four semantic nodes, ending with the write half of attention.
/// The positioned activation remains observable by the separate read/softmax.
fn ingress_graph(
  projected_observable~ : Bool,
  normalized_observable~ : Bool,
) -> @fusion_regions.Graph raise @fusion_regions.FusionRegionError {
  @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=projected_observable, effect=Pure),
    @fusion_regions.Node::new(
      [0],
      observable=normalized_observable,
      effect=Pure,
    ),
    @fusion_regions.Node::new([1], observable=true, effect=Pure),
    @fusion_regions.Node::new([2], observable=false, effect=Ordered),
  ])
}

///|
fn ingress_cut(
  schedule : @luna_fusion_strategy.FusionSchedule,
  qkv_operation~ : Int,
  projected_observable~ : Bool,
  normalized_observable~ : Bool,
) -> AttentionIngressCutPlan raise FusionPlanError {
  try {
    let graph = ingress_graph(projected_observable~, normalized_observable~)
    // These are supplied executable catalog regions, not a claim that every
    // mathematically legal fusion already has a backend implementation.
    let regions = match schedule {
      FullyFused => [@fusion_regions.Region::new(graph, start=0, count=4, [2])]
      ProducerSeparated =>
        [
          @fusion_regions.Region::new(graph, start=0, count=1, [0]),
          @fusion_regions.Region::new(graph, start=1, count=3, [2]),
        ]
      Unfused =>
        [
          @fusion_regions.Region::new(graph, start=0, count=1, [0]),
          @fusion_regions.Region::new(graph, start=1, count=1, [1]),
          @fusion_regions.Region::new(graph, start=2, count=1, [2]),
          @fusion_regions.Region::new(graph, start=3, count=1, []),
        ]
    }
    {
      schedule,
      qkv_operation,
      partition: @fusion_regions.Partition::new(graph, regions),
    }
  } catch {
    _ => raise InvalidFusionPlan(Dataflow)
  }
}

///|
pub fn AttentionIngressCutPlan::partition(
  self : AttentionIngressCutPlan,
) -> @fusion_regions.Partition {
  self.partition
}

///|
/// Compare complete ingress spans, never a fused kernel against only one
/// component of the unfused chain. The generic selector validates matching
/// graphs, cost bases, resources and deterministic stable catalog IDs.
pub fn select_attention_ingress_cut(
  alternatives : ArrayView[@luna_fusion_strategy.FusionAlternative],
  qkv_operation~ : Int,
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
) -> AttentionIngressCutPlan raise FusionPlanError {
  select_ingress_regions(
    alternatives,
    qkv_operation~,
    projected_observable=false,
    normalized_observable=false,
    maximum_workspace_bytes~,
    maximum_local_bytes~,
  )
}

///|
/// Model-plan consumers use the actual demand graph, filtering unavailable
/// elisions before ranking. Sharing never turns into a hidden recomputation.
pub fn QkvQkNormPositionedRopePagedKvWritePlan::select_cut(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
  alternatives : ArrayView[@luna_fusion_strategy.FusionAlternative],
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
) -> AttentionIngressCutPlan raise FusionPlanError {
  select_ingress_regions(
    alternatives,
    qkv_operation=self.qkv_operation.as_int(),
    projected_observable=!self.projected_exclusive,
    normalized_observable=!self.normalized_exclusive,
    maximum_workspace_bytes~,
    maximum_local_bytes~,
  )
}

///|
fn select_ingress_regions(
  alternatives : ArrayView[@luna_fusion_strategy.FusionAlternative],
  qkv_operation~ : Int,
  projected_observable~ : Bool,
  normalized_observable~ : Bool,
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
) -> AttentionIngressCutPlan raise FusionPlanError {
  if qkv_operation < 0 || qkv_operation > 2147483644 {
    raise InvalidFusionPlan(OperationOrder)
  }
  let candidates = []
  for index, alternative in alternatives {
    for prior in 0..<index {
      if alternative.schedule() == alternatives[prior].schedule() ||
        alternative.cost().region_cost().basis() !=
        alternatives[prior].cost().region_cost().basis() {
        raise InvalidFusionPlan(InvalidSelection)
      }
    }
    let cut = ingress_cut(
      alternative.schedule(),
      qkv_operation~,
      projected_observable~,
      normalized_observable~,
    ) catch {
      InvalidFusionPlan(Dataflow) => continue
      error => raise error
    }
    let (workspace_bytes, local_bytes) = alternative.resources()
    candidates.push(
      @fusion_regions.PartitionAlternative::new(
        id=match alternative.schedule() {
          Unfused => 0
          ProducerSeparated => 1
          FullyFused => 2
        },
        cut.partition,
        alternative.cost().region_cost(),
        workspace_bytes~,
        local_bytes~,
      ) catch {
        _ => raise InvalidFusionPlan(InvalidSelection)
      },
    )
  }
  let selected = @fusion_regions.select_partition(
    candidates,
    maximum_workspace_bytes~,
    maximum_local_bytes~,
  ) catch {
    @fusion_regions.NoFeasiblePartition =>
      raise InvalidFusionPlan(NoFeasibleCut)
    _ => raise InvalidFusionPlan(InvalidSelection)
  }
  ingress_cut(
    match selected {
      0 => Unfused
      1 => ProducerSeparated
      _ => FullyFused
    },
    qkv_operation~,
    projected_observable~,
    normalized_observable~,
  )
}

///|
pub fn AttentionIngressCutPlan::schedule(
  self : AttentionIngressCutPlan,
) -> @luna_fusion_strategy.FusionSchedule {
  self.schedule
}

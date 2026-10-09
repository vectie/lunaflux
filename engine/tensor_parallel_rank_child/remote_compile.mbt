///|
/// Startup-only reconstruction from independently approved local model bytes.
/// Both shard descriptions are inspected; only the selected rank is uploaded.
pub fn compile_remote_plan(
  model_root : @approved_fs.ApprovedRoot,
  source : @worker_wire.EncodedBootstrapSource,
  startup : @worker_wire.WorkerStartupContract,
  topology : @network_topology.TwoNodeTopology,
  budgets : ArrayView[@two_node_plan.RankMemoryBudget],
) -> @two_node_plan.TwoNodePlan raise {
  if budgets.length() != 2 ||
    startup.bootstrap_source_digest() != source.digest() {
    raise Invalid(SourcePolicy)
  }
  let (metadata, model) = load_rank_child_model(model_root, startup, source)
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    source.model().weights_locator(),
  )
  let inspections = [0, 1].map(rank => {
    let ceiling = budgets[rank]
      .runtime_ceiling_bytes()
      .min(source.ceilings().max_weight_arena_bytes())
    @device_materialize.inspect_sharded_file(
      model_root,
      metadata,
      model,
      locator,
      rank~,
      world_size=2,
      alignment_bytes=RANK_CHILD_ALIGNMENT_BYTES,
      max_arena_bytes=ceiling,
      limits=rank_child_weight_limits(
        source,
        model.operations().length(),
        ceiling,
      ),
    )
  })
  @two_node_plan.compile(
    model,
    topology,
    [inspections[0].rank_plan(), inspections[1].rank_plan()],
    budgets,
    tokens_per_page=source.execution().tokens_per_page(),
    maximum_pages=source.execution().total_page_count(),
  )
}

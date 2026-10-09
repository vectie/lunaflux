///|
test "network rank projections preserve ordinal zero and canonical sharded KV" {
  let (model, bindings) = tp_fixture(hidden=128, intermediate=128)
  let nodes = topology(memory=8388608L)
  let ranks = [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)]
  let compiled = @two_node_plan.compile(
    model,
    nodes,
    ranks,
    [budget(kv=4096L, ceiling=4194304L), budget(kv=2048L, ceiling=4194304L)],
    tokens_per_page=4,
    maximum_pages=16,
  )
  let kv = @tensor_parallel_kv_plan.admit_from_two_node_plan(
    model,
    @worker_protocol.ModelPlanGeneration::from_wire(7UL),
    compiled,
  )
  assert_eq(compiled.page_count(), 4)
  for rank in 0..<2 {
    let device = @tensor_parallel_device_plan.TensorParallelRankDevice::from_network_topology(
      nodes,
      rank~,
    )
    assert_eq(device.rank(), rank)
    assert_eq(device.world_size(), 2)
    assert_eq(device.device_ordinal(), 0)
    assert_eq(kv.rank_layout(rank), compiled.kv_layout())
    let binding = kv.rank_binding(rank)
    assert_eq(binding.rank(), rank)
    assert_eq(binding.process_ordinal(), 0)
    assert_eq(binding.generation_value(), 7UL)
    assert_eq(binding.local_kv_head_count(), 1)
    assert_eq(binding.arena_bytes(), 2048L)
    assert_eq(
      binding.kv_reservation_bytes(),
      compiled.rank_budget(rank).unwrap().kv_bytes(),
    )
  }
  assert_not_eq(kv.rank_binding(0).digest(), kv.rank_binding(1).digest())
  let (other, _) = tp_fixture(
    hidden=128,
    intermediate=128,
    content="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
  try
    @tensor_parallel_kv_plan.admit_from_two_node_plan(
      other,
      @worker_protocol.ModelPlanGeneration::from_wire(7UL),
      compiled,
    )
  catch {
    @tensor_parallel_kv_plan.RejectedTensorParallelKvPlan(
      rule=ModelIdentity,
      ..
    ) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("foreign model admitted")
  }
  try
    @tensor_parallel_device_plan.TensorParallelRankDevice::from_network_topology(
      nodes,
      rank=2,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("foreign rank admitted")
  }
}

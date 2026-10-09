///|
fn topology(
  memory? : Int64 = 1048576L,
) -> @network_topology.TwoNodeTopology raise {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let a = @network_topology.NetworkRank::new(
    node=@network_topology.NodeIdentity::new("spark-a"),
    device_ordinal=0,
    target~,
    physical_memory_bytes=memory,
    rdma_interface="mlx5_0",
  )
  let b = @network_topology.NetworkRank::new(
    node=@network_topology.NodeIdentity::new("spark-b"),
    device_ordinal=0,
    target~,
    physical_memory_bytes=memory,
    rdma_interface="mlx5_0",
  )
  @network_topology.TwoNodeTopology::new(a, b)
}

///|
fn budget(
  kv? : Int64 = 1024L,
  ceiling? : Int64 = 100000L,
  workspace? : Int64 = 1024L,
  reserve? : Int64 = 4096L,
) -> @two_node_plan.RankMemoryBudget raise {
  @two_node_plan.RankMemoryBudget::new(
    system_reserved_bytes=reserve,
    runtime_ceiling_bytes=ceiling,
    activation_workspace_bytes=workspace,
    collective_bytes=1024L,
    kv_bytes=kv,
  )
}

///|
test "existing family lowering compiles deterministically for two separate nodes" {
  let (model, bindings) = tp_fixture()
  let ranks = [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)]
  let budgets = [budget(kv=1024L), budget(kv=640L)]
  let nodes = topology()
  let compiled = @two_node_plan.compile(
    model,
    nodes,
    ranks,
    budgets,
    tokens_per_page=4,
    maximum_pages=100,
  )
  assert_eq(
    compiled,
    @two_node_plan.compile(
      model,
      nodes,
      ranks,
      budgets,
      tokens_per_page=4,
      maximum_pages=100,
    ),
  )
  // Each K/V component has 16 logical bytes but reserves 256 aligned bytes.
  assert_eq(compiled.kv_layout().page_logical_bytes(), 16L)
  assert_eq(compiled.rank_capacity(0).unwrap().kv_page_bytes(), 512L)
  assert_eq(compiled.rank_capacity(0).unwrap().maximum_kv_pages(), 2L)
  assert_eq(compiled.rank_capacity(1).unwrap().maximum_kv_pages(), 1L)
  assert_eq(compiled.page_count(), 1)
  assert_eq(compiled.kv_layout().arena_bytes(), 512L)
  assert_eq(
    compiled.rank_capacity(0).unwrap().weight_bytes(),
    ranks[0].arena_bytes(),
  )
  assert_eq(compiled.rank_plan(1), Some(ranks[1]))
  assert_eq(compiled.rank_capacity(2), None)
  assert_eq(compiled.rank_budget(0), Some(budgets[0]))
  // Mutating input arrays cannot mutate the compiled snapshot.
  ranks[0] = ranks[1]
  budgets[0] = budget(kv=32L)
  assert_eq(compiled.rank_plan(0).unwrap().topology().rank(), 0)
  assert_eq(compiled.page_count(), 1)
}

///|
test "page ceiling caps capacity without arithmetic narrowing overflow" {
  let (model, bindings) = tp_fixture()
  let compiled = @two_node_plan.compile(
    model,
    topology(),
    [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)],
    [budget(kv=4096L), budget(kv=4096L)],
    tokens_per_page=4,
    maximum_pages=3,
  )
  assert_eq(compiled.page_count(), 3)
}

///|
test "rank order world size and model identity are explicit" {
  let (model, bindings) = tp_fixture()
  let ranks = [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)]
  for invalid in [[ranks[1], ranks[0]], [ranks[0], ranks[0]], [ranks[0]]] {
    try
      @two_node_plan.compile(
        model,
        topology(),
        invalid,
        [budget(), budget()],
        tokens_per_page=4,
        maximum_pages=100,
      )
    catch {
      @two_node_plan.Rejected(rule=RankOrder, ..) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("invalid rank binding accepted")
    }
  }
  let (other, _) = tp_fixture(
    content="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
  try
    @two_node_plan.compile(
      other,
      topology(),
      ranks,
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=ModelIdentity, rank=0) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("foreign model accepted")
  }
}

///|
test "aggregate spare memory cannot cover an overfull rank" {
  let (model, bindings) = tp_fixture()
  let ranks = [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)]
  try
    @two_node_plan.compile(
      model,
      topology(),
      ranks,
      [budget(ceiling=900000L), budget(ceiling=2048L)],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=RankCapacity, rank=1) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("pooled memory admitted")
  }
}

///|
test "unified memory retains system headroom" {
  let (model, bindings) = tp_fixture()
  try
    @two_node_plan.compile(
      model,
      topology(memory=100000L),
      [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)],
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=MemoryBudget, rank=0) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("system reserve ignored")
  }
}

///|
test "less than one KV page cannot admit a group" {
  let (model, bindings) = tp_fixture()
  try
    @two_node_plan.compile(
      model,
      topology(),
      [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)],
      [budget(kv=511L), budget()],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=RankCapacity, rank=0) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("zero pages admitted")
  }
}

///|
test "reservation arithmetic fails before signed overflow" {
  let (model, bindings) = tp_fixture()
  try
    @two_node_plan.compile(
      model,
      topology(memory=9223372036854775807L),
      [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)],
      [
        budget(ceiling=9223372036854770000L, workspace=9223372036854775807L),
        budget(),
      ],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=Overflow, rank=0) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("overflow admitted")
  }
}

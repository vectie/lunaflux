///|
fn kvp_identity(
  content? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @spec.ModelIdentity {
  try! @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(content),
    try! @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn kvp_geometry(
  heads? : Int = 8,
  layers? : Int = 2,
  width? : Int = 64,
) -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=layers,
    kv_head_count=heads,
    head_dimension=width,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
fn kvp_target(minor? : Int = 0) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn kvp_generation(
  value? : UInt64 = 7UL,
) -> @worker_protocol.ModelPlanGeneration {
  try! @worker_protocol.ModelPlanGeneration::from_wire(value)
}

///|
fn kvp_device_facts(
  identity : @spec.ModelIdentity,
  world : Int,
  changed_rank? : Int = -1,
  changed_world? : Int = -1,
  changed_identity_rank? : Int = -1,
  changed_target_rank? : Int = -1,
  changed_ordinal_rank? : Int = -1,
) -> Array[RankDeviceFacts] {
  let output : Array[RankDeviceFacts] = []
  for rank in 0..<world {
    output.push({
      identity: if rank == changed_identity_rank {
        kvp_identity(
          content="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        )
      } else {
        identity
      },
      rank: if rank == changed_rank {
        0
      } else {
        rank
      },
      world_size: if rank == changed_world {
        world + 1
      } else {
        world
      },
      device_ordinal: if rank == changed_ordinal_rank {
        9
      } else {
        rank * 3 + 2
      },
      target: kvp_target(minor=if rank == changed_target_rank { 1 } else { 0 }),
    })
  }
  output
}

///|
fn kvp_runtime_facts(
  world : Int,
  changed_rank? : Int = -1,
  changed_world? : Int = -1,
  changed_target_rank? : Int = -1,
  reservation? : Int64 = 1048576L,
) -> Array[RankRuntimeFacts] {
  let output : Array[RankRuntimeFacts] = []
  for rank in 0..<world {
    output.push({
      rank: if rank == changed_rank {
        0
      } else {
        rank
      },
      world_size: if rank == changed_world {
        world + 1
      } else {
        world
      },
      process_ordinal: rank * 3 + 2,
      target: kvp_target(minor=if rank == changed_target_rank { 1 } else { 0 }),
      kv_reserved_bytes: reservation,
    })
  }
  output
}

///|
fn kvp_admit(
  geometry? : @plan.KvCacheGeometry = kvp_geometry(),
  generation? : UInt64 = 7UL,
  world? : Int = 2,
  devices? : Array[RankDeviceFacts],
  runtime_ranks? : Array[RankRuntimeFacts],
  runtime_target? : @catalog.DeviceTarget = kvp_target(),
  tokens? : Int = 16,
  pages? : Int = 4,
) -> TensorParallelKvPlan raise TensorParallelKvPlanError {
  let identity = kvp_identity()
  let actual_devices = match devices {
    Some(value) => value
    None => kvp_device_facts(identity, world)
  }
  let actual_runtime = match runtime_ranks {
    Some(value) => value
    None => kvp_runtime_facts(world)
  }
  admit_facts(
    identity, geometry, generation, actual_devices, runtime_target, actual_runtime,
    tokens, pages,
  )
}

///|
fn kvp_expect(
  body : () -> Unit raise TensorParallelKvPlanError,
  expected : TensorParallelKvPlanRule,
) -> Unit raise {
  try body() catch {
    RejectedTensorParallelKvPlan(rule~, rank=_) => assert_eq(rule, expected)
  } noraise {
    _ => fail("invalid rank-local KV plan was admitted")
  }
}

///|
test "rank-local plan derives exact sharded geometry and output bindings" {
  let plan = kvp_admit()
  assert_eq(plan.generation_value(), 7UL)
  assert_eq(plan.world_size(), 2)
  assert_eq(plan.global_kv_head_count(), 8)
  assert_eq(plan.local_kv_head_count(), 4)
  assert_eq(plan.layer_count(), 2)
  assert_eq(plan.head_dimension(), 64)
  assert_eq(plan.layout_version(), 1)
  assert_eq(plan.tokens_per_page(), 16)
  assert_eq(plan.total_page_count(), 4)
  assert_eq(plan.page_logical_bytes(), 8192L)
  assert_eq(plan.rank_arena_bytes(), 131072L)
  assert_eq(plan.full_arena_bytes(), 262144L)
  assert_eq(plan.rank_kv_reservation_bytes(1), 1048576L)
  let rank = plan.rank_binding(1)
  assert_eq(rank.identity(), plan.identity())
  assert_eq(rank.generation_value(), 7UL)
  assert_eq(rank.rank(), 1)
  assert_eq(rank.world_size(), 2)
  assert_eq(rank.process_ordinal(), 5)
  assert_eq(rank.local_kv_head_count(), 4)
  assert_eq(rank.arena_bytes(), 131072L)
  assert_eq(rank.kv_reservation_bytes(), 1048576L)
  assert_eq(rank.digest().as_sha256().length(), 64)
  assert_false(plan.rank_binding(0).digest() == rank.digest())
}

///|
test "every rank receives the same canonical immutable local layout" {
  let plan = kvp_admit()
  let rank_zero = plan.rank_layout(0)
  let rank_one = plan.rank_layout(1)
  assert_eq(rank_zero, rank_one)
  assert_eq(rank_zero.identity(), plan.identity())
  assert_eq(rank_zero.geometry().kv_head_count(), plan.local_kv_head_count())
  assert_eq(rank_zero.page_stride_bytes(), plan.page_stride_bytes())
  assert_eq(rank_zero.arena_bytes(), plan.rank_arena_bytes())
  kvp_expect(() => ignore(plan.rank_layout(-1)), Rank)
  kvp_expect(() => ignore(plan.rank_layout(2)), Rank)
}

///|
test "admission rejects rank world model target ordinal and runtime substitution" {
  let identity = kvp_identity()
  kvp_expect(
    () => {
      ignore(kvp_admit(devices=kvp_device_facts(identity, 2, changed_rank=1)))
    },
    Rank,
  )
  kvp_expect(
    () => {
      ignore(kvp_admit(devices=kvp_device_facts(identity, 2, changed_world=1)))
    },
    WorldSize,
  )
  kvp_expect(
    () => {
      ignore(
        kvp_admit(
          devices=kvp_device_facts(identity, 2, changed_identity_rank=1),
        ),
      )
    },
    ModelIdentity,
  )
  kvp_expect(
    () => {
      ignore(
        kvp_admit(devices=kvp_device_facts(identity, 2, changed_target_rank=1)),
      )
    },
    DeviceTarget,
  )
  kvp_expect(
    () => {
      ignore(
        kvp_admit(devices=kvp_device_facts(identity, 2, changed_ordinal_rank=1)),
      )
    },
    DeviceOrdinal,
  )
  kvp_expect(
    () => ignore(kvp_admit(runtime_ranks=kvp_runtime_facts(2, changed_world=1))),
    WorldSize,
  )
  kvp_expect(
    () => {
      ignore(
        kvp_admit(runtime_ranks=kvp_runtime_facts(2, changed_target_rank=1)),
      )
    },
    DeviceTarget,
  )
}

///|
test "admission rejects indivisible heads ceilings overflow and aligned replication" {
  kvp_expect(
    () => ignore(kvp_admit(geometry=kvp_geometry(heads=3))),
    KvHeadDivisibility,
  )
  kvp_expect(
    () => {
      ignore(kvp_admit(runtime_ranks=kvp_runtime_facts(2, reservation=131071L)))
    },
    RankArenaCeiling,
  )
  kvp_expect(
    () => ignore(kvp_admit(tokens=2147483647, pages=2147483647)),
    SizeOverflow,
  )
  kvp_expect(
    () => {
      ignore(
        kvp_admit(
          geometry=kvp_geometry(heads=2, layers=1, width=1),
          tokens=1,
          pages=1,
        ),
      )
    },
    FullArenaReplication,
  )
}

///|
test "current generation authentication rejects stale and substituted values" {
  let plan = kvp_admit()
  plan.require_current_contract(kvp_identity(), kvp_generation(), 1)
  kvp_expect(
    () => plan.require_current_generation(kvp_generation(value=8UL)),
    ModelGeneration,
  )
  kvp_expect(
    () => {
      plan.require_current_contract(
        kvp_identity(
          content="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        ),
        kvp_generation(),
        1,
      )
    },
    ModelIdentity,
  )
  kvp_expect(
    () => plan.require_current_contract(kvp_identity(), kvp_generation(), 2),
    LayoutVersion,
  )
}

///|
fn generic_test_content() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn generic_test_model() -> @plan.ModelPlan raise {
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  @plan.ModelPlan::new_stateless(
    content_digest=generic_test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=8,
      attention_head_count=2,
      kv_head_count=2,
      head_dimension=2,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=2,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[operation],
    final_output=@plan.ActivationRef::new(0),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema([operation]),
  )
}

///|
fn generic_row_tensor(
  reference : Int,
  full_rows : Int64,
  row_start : Int64,
  row_count : Int64,
  offset : Int64,
  byte_length : Int64,
) -> TensorRankPlacement raise {
  let tensor = @plan.TensorRef::new(reference)
  let region = TensorArenaRegion::new(
    offset_bytes=offset,
    byte_length~,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  let extent = TensorShardExtent::new(
    full_rows~,
    full_columns=2L,
    row_start~,
    row_count~,
    column_start=0L,
    column_count=2L,
    byte_length~,
  )
  let recipe = TensorTransferRecipe::new(
    reference=tensor,
    source_base_bytes=0L,
    source_stride_bytes=0L,
    destination_base_bytes=offset,
    destination_stride_bytes=0L,
    segment_bytes=byte_length,
    segment_count=1L,
    destination_region=region,
  )
  TensorRankPlacement::new(
    reference=tensor,
    placement=ShardRows,
    extent~,
    region~,
    transfer_recipe=recipe,
  )
}

///|
fn generic_operation(
  index? : Int = 0,
  output_width? : Int = 4,
) -> TensorParallelOperationPlan raise {
  TensorParallelOperationPlan::new(
    operation=@plan.OperationId::new(index),
    kind=Embedding,
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    input_width=1,
    output_width~,
    inner_width=0,
    query_head_count=0,
    kv_head_count=0,
    head_dimension=0,
  )
}

///|
test "generic plan rejects shard world rank and divisibility transplant" {
  let model = generic_test_model()
  let world_two = TensorParallelTopology::new(rank=0, world_size=2)
  let source = generic_row_tensor(0, 8L, 0L, 4L, 0L, 16L)
  for
    substituted in [
      TensorParallelTopology::new(rank=0, world_size=4),
      TensorParallelTopology::new(rank=1, world_size=2),
    ] {
    try
      ignore(
        TensorParallelRankPlan::new(
          model~,
          topology=substituted,
          alignment_bytes=16L,
          arena_bytes=16L,
          materialized_bytes=16L,
          tensors=[source],
          operations=[generic_operation()],
          collectives=[],
        ),
      )
    catch {
      InvalidTensorParallelPlan(rule=Extent, ..) => ()
      _ => fail("unexpected shard topology substitution error")
    } noraise {
      _ => fail("shard topology substitution was admitted")
    }
  }
  let indivisible = generic_row_tensor(0, 7L, 0L, 3L, 0L, 16L)
  try
    ignore(
      TensorParallelRankPlan::new(
        model~,
        topology=world_two,
        alignment_bytes=16L,
        arena_bytes=16L,
        materialized_bytes=16L,
        tensors=[indivisible],
        operations=[generic_operation()],
        collectives=[],
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=Extent, ..) => ()
    _ => fail("unexpected indivisible shard error")
  } noraise {
    _ => fail("indivisible shard was admitted")
  }
}

///|
test "generic plan rejects all gather built for another world" {
  let model = generic_test_model()
  let topology = TensorParallelTopology::new(rank=0, world_size=4)
  let tensor = generic_row_tensor(0, 8L, 0L, 2L, 0L, 8L)
  let site = TensorCollectiveSite::new(
    sequence=1,
    operation=@plan.OperationId::new(0),
    kind=LastAxisAllGather,
    full_width=8,
    local_width=4,
    world_size=2,
  )
  try
    ignore(
      TensorParallelRankPlan::new(
        model~,
        topology~,
        alignment_bytes=16L,
        arena_bytes=8L,
        materialized_bytes=8L,
        tensors=[tensor],
        operations=[generic_operation(output_width=4)],
        collectives=[site],
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=CollectiveGeometry, ..) => ()
    _ => fail("unexpected all-gather world substitution error")
  } noraise {
    _ => fail("all-gather from another world was admitted")
  }
}

///|
test "generic plan rejects overlap stale order and arena accounting" {
  let model = generic_test_model()
  let topology = TensorParallelTopology::new(rank=0, world_size=2)
  let first = generic_row_tensor(0, 8L, 0L, 4L, 0L, 16L)
  let overlap = generic_row_tensor(1, 8L, 0L, 4L, 0L, 16L)
  try
    ignore(
      TensorParallelRankPlan::new(
        model~,
        topology~,
        alignment_bytes=16L,
        arena_bytes=32L,
        materialized_bytes=32L,
        tensors=[first, overlap],
        operations=[generic_operation()],
        collectives=[],
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=TensorOrder, ..) => ()
    _ => fail("unexpected overlap error")
  } noraise {
    _ => fail("overlapping tensor regions were admitted")
  }
  let stale = generic_row_tensor(1, 8L, 0L, 4L, 0L, 16L)
  try
    ignore(
      TensorParallelRankPlan::new(
        model~,
        topology~,
        alignment_bytes=16L,
        arena_bytes=16L,
        materialized_bytes=16L,
        tensors=[stale],
        operations=[generic_operation()],
        collectives=[],
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=TensorOrder, ..) => ()
    _ => fail("unexpected stale tensor reference error")
  } noraise {
    _ => fail("stale tensor reference was admitted")
  }
  for accounting in [(32L, 16L), (16L, 8L)] {
    try
      ignore(
        TensorParallelRankPlan::new(
          model~,
          topology~,
          alignment_bytes=16L,
          arena_bytes=accounting.0,
          materialized_bytes=accounting.1,
          tensors=[first],
          operations=[generic_operation()],
          collectives=[],
        ),
      )
    catch {
      InvalidTensorParallelPlan(rule=MaterializedBytes, ..) => ()
      _ => fail("unexpected arena accounting error")
    } noraise {
      _ => fail("incorrect arena accounting was admitted")
    }
  }
}

///|
test "generic transfer recipe rejects overflow and segment order" {
  let tensor = @plan.TensorRef::new(0)
  let region = TensorArenaRegion::new(
    offset_bytes=0L,
    byte_length=16L,
    alignment_bytes=16L,
    max_arena_bytes=16L,
  )
  try
    ignore(
      TensorTransferRecipe::new(
        reference=tensor,
        source_base_bytes=9223372036854775800L,
        source_stride_bytes=0L,
        destination_base_bytes=0L,
        destination_stride_bytes=0L,
        segment_bytes=16L,
        segment_count=1L,
        destination_region=region,
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=ArithmeticOverflow, ..) => ()
    _ => fail("unexpected transfer overflow error")
  } noraise {
    _ => fail("transfer overflow was admitted")
  }
  try
    ignore(
      TensorTransferRecipe::new(
        reference=tensor,
        source_base_bytes=0L,
        source_stride_bytes=8L,
        destination_base_bytes=0L,
        destination_stride_bytes=4L,
        segment_bytes=8L,
        segment_count=2L,
        destination_region=region,
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=TransferRecipe, ..) => ()
    _ => fail("unexpected transfer segment order error")
  } noraise {
    _ => fail("overlapping transfer segments were admitted")
  }
}

///|
test "generic arena regions reject weak alignment and misaligned offsets" {
  for pair in [(0L, 8L), (8L, 16L)] {
    try
      ignore(
        TensorArenaRegion::new(
          offset_bytes=pair.0,
          byte_length=16L,
          alignment_bytes=pair.1,
          max_arena_bytes=64L,
        ),
      )
    catch {
      InvalidTensorParallelPlan(rule=Region, ..) => ()
      _ => fail("unexpected arena-alignment rejection")
    } noraise {
      _ => fail("invalid arena alignment was admitted")
    }
  }
}

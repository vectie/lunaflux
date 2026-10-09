///|
fn rewrite_rank(
  model : @plan.ModelPlan,
  rank : @tensor_parallel_plan.TensorParallelRankPlan,
  source_shift? : Int64 = 0L,
  drop_collective? : Bool = false,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise {
  let tensors : Array[@tensor_parallel_plan.TensorRankPlacement] = []
  for index in 0..<rank.tensor_count() {
    let tensor = rank.tensor(@plan.TensorRef::new(index)).unwrap()
    let recipe = tensor.transfer_recipe()
    let count = recipe.segment_count()
    let first = recipe.segment(0L)
    let last = recipe.segment(count - 1L)
    let source_stride = if count > 1L {
      (last.source_payload_offset_bytes() - first.source_payload_offset_bytes()) /
      (count - 1L)
    } else {
      0L
    }
    let destination_stride = if count > 1L {
      (last.destination_offset_bytes() - first.destination_offset_bytes()) /
      (count - 1L)
    } else {
      0L
    }
    tensors.push(
      @tensor_parallel_plan.TensorRankPlacement::new(
        reference=tensor.reference(),
        placement=tensor.placement(),
        extent=tensor.extent(),
        region=tensor.region(),
        transfer_recipe=@tensor_parallel_plan.TensorTransferRecipe::new(
          reference=tensor.reference(),
          source_base_bytes=first.source_payload_offset_bytes() + source_shift,
          source_stride_bytes=source_stride,
          destination_base_bytes=first.destination_offset_bytes(),
          destination_stride_bytes=destination_stride,
          segment_bytes=recipe.segment_byte_length(),
          segment_count=count,
          destination_region=tensor.region(),
        ),
      ),
    )
  }
  let collective_count = rank.collective_count() -
    (if drop_collective { 1 } else { 0 })
  @tensor_parallel_plan.TensorParallelRankPlan::new(
    model~,
    topology=rank.topology(),
    alignment_bytes=rank.alignment_bytes(),
    arena_bytes=rank.arena_bytes(),
    materialized_bytes=rank.materialized_bytes(),
    tensors~,
    operations=[
      for operation in model.operations() => {
        rank.operation(operation.id()).unwrap()
      }
    ],
    collectives=[
      for index in 0..<collective_count => rank.collective(index).unwrap()
    ],
  )
}

///|
test "matching dimensions cannot disguise different weight source tensors" {
  let (model, bindings) = tp_fixture()
  let left = tp_plan(model, bindings, 0, 2)
  let right = rewrite_rank(
    model,
    tp_plan(model, bindings, 1, 2),
    source_shift=16L,
  )
  try
    @two_node_plan.compile(
      model,
      topology(),
      [left, right],
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=TensorPartition, ..) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("source substitution accepted")
  }
}

///|
test "different collective schedules cannot enter one group" {
  let (model, bindings) = tp_fixture()
  let left = tp_plan(model, bindings, 0, 2)
  let right = rewrite_rank(
    model,
    tp_plan(model, bindings, 1, 2),
    drop_collective=true,
  )
  try
    @two_node_plan.compile(
      model,
      topology(),
      [left, right],
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=100,
    )
  catch {
    @two_node_plan.Rejected(rule=CollectiveMismatch, ..) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("collective substitution accepted")
  }
}

///|
test "plan digest binds budgets topology source offsets and effective page count" {
  let (model, bindings) = tp_fixture()
  let ranks = [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)]
  let nodes = topology()
  let baseline = @two_node_plan.compile(
    model,
    nodes,
    ranks,
    [budget(), budget()],
    tokens_per_page=4,
    maximum_pages=10,
  ).digest()
  let alternatives = [
    @two_node_plan.compile(
      model,
      nodes,
      ranks,
      [budget(workspace=2048L), budget()],
      tokens_per_page=4,
      maximum_pages=10,
    ),
    @two_node_plan.compile(
      model,
      topology(memory=2000000L),
      ranks,
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=10,
    ),
    @two_node_plan.compile(
      model,
      nodes,
      [
        rewrite_rank(model, ranks[0], source_shift=16L),
        rewrite_rank(model, ranks[1], source_shift=16L),
      ],
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=10,
    ),
    @two_node_plan.compile(
      model,
      nodes,
      ranks,
      [budget(), budget()],
      tokens_per_page=4,
      maximum_pages=1,
    ),
  ]
  assert_eq(baseline.as_sha256().length(), 64)
  for alternative in alternatives {
    assert_true(alternative.digest() != baseline)
  }
}

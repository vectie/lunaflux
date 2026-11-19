///|
fn expert_operand(
  rows : Int32,
  columns : Int32,
) -> @precision_ir.OperandPlan raise {
  @precision_ir.plan_operand(
    role=Weight,
    representation=@format.Representation::new(format=NvFp4, rows~, columns~),
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
    memory_ceiling=100000000L,
    tile_elements=32,
  )
}

///|
test "BF16 each-add expert combination preserves its round and rejects partial ownership" {
  let make = fn(owned, publication) raise {
    @precision_ir.ExpertMlpPrecision::new(
      gate=expert_operand(32, 64),
      up=expert_operand(32, 64),
      down=expert_operand(64, 32),
      expert_count=3,
      local_experts=owned,
      selected_experts=2,
      maximum_rows=2,
      down_publication=publication,
      combine_rounding=Bf16EachAdd,
      memory_ceiling=1000000L,
    )
  }
  let plan = make([2, 0, 1], LocalBFloat16)
  assert_true(
    plan.expression(CombineUpdate).root()
    is Convert(
      Float32,
      Convert(BFloat16, Binary(AddRN, Input(0), Convert(Float32, Input(1))))
    ),
  )
  assert_true(
    try make([0, 1], LocalBFloat16) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try make([0, 1, 2], ReduceFloat32BeforeRound) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @precision_ir.ExpertTensorPartition::new(
        plan,
        rank=0,
        world_size=2,
        memory_ceiling=1000000L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
test "compact expert plan preserves noncontiguous rank placement and bounded scratch" {
  let plan = @precision_ir.ExpertMlpPrecision::new(
    gate=expert_operand(32, 64),
    up=expert_operand(32, 64),
    down=expert_operand(64, 32),
    expert_count=7,
    local_experts=[5, 1],
    selected_experts=2,
    maximum_rows=3,
    clamp=10.0,
    memory_ceiling=100000L,
  )
  assert_eq(plan.geometry(), (3, 64, 32, 7, 2))
  assert_eq(Array::makei(7, i => plan.expert_map()[i]), [
    -1, 1, -1, -1, -1, 0, -1,
  ])
  assert_eq(Array::makei(3, i => plan.weight_offsets()[i]), [0L, 1168L, 2336L])
  assert_eq(plan.expert_stride(), 3504L)
  assert_eq(plan.bank_bytes(), 7008L)
  assert_eq(plan.workspace_bytes(), (384L, 768L))
  assert_eq(plan.total_bytes(), 9408L)
  assert_eq(plan.clamp(), Some(10.0))
}

///|
test "expert plan rejects expansion, incompatible projections and duplicate placement" {
  let gate = expert_operand(32, 64)
  let up = expert_operand(32, 64)
  let down = expert_operand(64, 32)
  try
    @precision_ir.ExpertMlpPrecision::new(
      gate~,
      up~,
      down~,
      expert_count=7,
      local_experts=[5, 5],
      selected_experts=2,
      maximum_rows=3,
      memory_ceiling=100000L,
    )
  catch {
    IllegalPlacement => ()
    _ => fail("wrong duplicate placement error")
  } noraise {
    _ => fail("duplicate expert bank accepted")
  }
  try
    @precision_ir.ExpertMlpPrecision::new(
      gate~,
      up~,
      down=gate,
      expert_count=7,
      local_experts=[5],
      selected_experts=2,
      maximum_rows=3,
      memory_ceiling=100000L,
    )
  catch {
    IllegalPlacement => ()
    _ => fail("wrong incompatible geometry error")
  } noraise {
    _ => fail("wrong down geometry accepted")
  }
  try
    @precision_ir.ExpertMlpPrecision::new(
      gate~,
      up~,
      down~,
      expert_count=7,
      local_experts=[5, 1],
      selected_experts=2,
      maximum_rows=3,
      memory_ceiling=9407L,
    )
  catch {
    MemoryBudgetExceeded => ()
    _ => fail("wrong workspace ceiling error")
  } noraise {
    _ => fail("workspace escaped ceiling")
  }
}

///|
test "packed layout aligns global scales without changing nibble payload" {
  let nv = @precision_ir.packed_buffer_layout(
    @format.Representation::new(format=NvFp4, rows=1, columns=16),
  )
  assert_eq(nv.scales_offset(), Some(8L))
  assert_eq(nv.global_scale_offset(), Some(12L))
  assert_eq(nv.buffer_bytes(), 16L)
  let mx = @precision_ir.packed_buffer_layout(
    @format.Representation::new(format=MxFp4, rows=1, columns=32),
  )
  assert_eq(mx.scales_offset(), Some(16L))
  assert_eq(mx.global_scale_offset(), None)
  assert_eq(mx.buffer_bytes(), 17L)
}

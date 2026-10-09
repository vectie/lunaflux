///|
extern "C" fn live_resources() -> Int = "lunaflux_device_worker_fake_live_children"

///|
fn program_fixture() -> (@moe.MoeProgram, Int64, Int64) raise {
  let operand = @precision.plan_operand(
    role=Weight,
    representation=@format.Representation::new(format=Bf16, rows=32, columns=32),
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    memory_ceiling=100000L,
    tile_elements=32,
  )
  let routed = @precision.ExpertMlpPrecision::new(
    gate=operand,
    up=operand,
    down=operand,
    expert_count=4,
    local_experts=[3, 1],
    selected_experts=2,
    maximum_rows=2,
    memory_ceiling=100000L,
  )
  let shared = @precision.ExpertMlpPrecision::new(
    gate=operand,
    up=operand,
    down=operand,
    expert_count=1,
    local_experts=[0],
    selected_experts=1,
    maximum_rows=2,
    memory_ceiling=100000L,
  )
  (
    @moe.MoeProgram::new(
      @source.packed_expert_mlp_source(routed, symbol="routed"),
      @source.packed_expert_mlp_source(shared, symbol="shared"),
      @combine.moe_combine_source(
        @precision.MoeCombinePrecision::new(
          maximum_rows=2,
          hidden=32,
          rounding=SumThenBf16,
          memory_ceiling=640L,
        ),
        symbol="finalize",
      ),
    ),
    routed.bank_bytes(),
    shared.bank_bytes(),
  )
}

///|
test "complete MoE Program prepares compact workspace and reuses seven launches" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let (program, routed_bytes, shared_bytes) = program_fixture()
  defer (program.abort() catch { _ => () })
  // Routed: map16 + activation256 + weighted256 + F32output256.
  // Shared: map4 + activation128 + weighted128 + F32output256 + unit routing16.
  assert_eq(program.workspace_bytes(), 1316L)
  assert_true(program.source_bytes().length() > 0)
  let external = [20L, 128L, 16L, 16L, routed_bytes, shared_bytes, 128L].map(bytes => {
    context.allocate(bytes)
  })
  defer (for allocation in external.rev() {
    allocation.close() catch {
      _ => ()
    }
  })
  program.prepare(
    context,
    aot,
    stream,
    maximum_workspace_bytes=1316L,
    counts=external[0],
    input=external[1],
    selected_indices=external[2],
    selected_weights=external[3],
    routed_bank=external[4],
    shared_bank=external[5],
    output=external[6],
  )
  let before = enqueues()
  begin_allocations()
  for sequence in 1UL..<17UL {
    program.submit(
      live_rows=if sequence % 2UL == 0UL { 2 } else { 1 },
      plan_sequence=sequence,
      global_sequence=sequence,
    )
    while !program.poll() {
      ()
    }
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 112UL)
  program.close()
  program.close()
  // External operands remain alive and owned by the caller.
  context.copy_from_fixed_host(external[6], [0, 0], 0L, 0L, 2L)
  for allocation in external.rev() {
    allocation.close()
  }
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "MoE Program budget refusal and failed binding release all partial owners" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let buffer = context.allocate(20L)
  defer (buffer.close() catch { _ => () })
  let baseline = live_resources()
  for budget in [1315L, 1316L] {
    let (program, _, _) = program_fixture()
    defer (program.close() catch { _ => () })
    try
      program.prepare(
        context,
        aot,
        stream,
        maximum_workspace_bytes=budget,
        counts=buffer,
        input=buffer,
        selected_indices=buffer,
        selected_weights=buffer,
        routed_bank=buffer,
        shared_bank=buffer,
        output=buffer,
      )
    catch {
      @device.DeviceError::InvalidArgument(_) => ()
      error => raise error
    } noraise {
      _ => abort("undersized operands must not bind")
    }
    if budget == 1315L {
      assert_eq(live_resources(), baseline)
    } else {
      assert_true(live_resources() > baseline)
    }
    program.close()
    assert_eq(live_resources(), baseline)
  }
}

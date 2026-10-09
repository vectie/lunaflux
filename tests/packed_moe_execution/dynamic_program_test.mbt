///|
fn dynamic_program_fixture() -> (@moe.MoeProgram, Int64, Int64) raise {
  fn expert(shared : Bool) -> @precision.ExpertMlpPrecision raise {
    let operand = @precision.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=if shared { MxFp8E4M3 } else { MxFp4 },
        rows=128,
        columns=128,
        grouping=if shared {
          Block(rows=128, columns=128)
        } else {
          Block(rows=1, columns=32)
        },
      ),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=1000000L,
      tile_elements=32,
    )
    fn quantization(rows : Int) -> @precision.QuantizationPlan raise {
      @precision.plan_operand(
        role=Activation,
        representation=@format.Representation::new(
          format=MxFp8E4M3,
          rows~,
          columns=128,
          grouping=Block(rows=1, columns=128),
        ),
        placement=QuantizeBeforeConsumer,
        supported_placements=[QuantizeBeforeConsumer],
        accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
        memory_ceiling=1000000L,
      )
      .quantization()
      .with_amax_floor(0.0001)
    }
    @precision.ExpertMlpPrecision::new(
      gate=operand,
      up=operand,
      down=operand,
      expert_count=if shared { 1 } else { 4 },
      local_experts=if shared { [0] } else { [3, 1] },
      selected_experts=if shared { 1 } else { 2 },
      maximum_rows=2,
      activation_rounding=F32Product,
      routing_placement=BeforeDownInputRound,
      activation_quantization=(
        quantization(2),
        quantization(if shared { 2 } else { 4 }),
      ),
      memory_ceiling=1000000L,
    )
  }
  let routed = expert(false)
  let shared = expert(true)
  (
    @moe.MoeProgram::new(
      @source.packed_expert_mlp_source(routed, symbol="routed_dynamic"),
      @source.packed_expert_mlp_source(shared, symbol="shared_dynamic"),
      @combine.moe_combine_source(
        @precision.MoeCombinePrecision::new(
          maximum_rows=2,
          hidden=128,
          rounding=Bf16RoutedThenSum,
          memory_ceiling=2560L,
        ),
        symbol="finalize_dynamic",
      ),
    ),
    routed.bank_bytes(),
    shared.bank_bytes(),
  )
}

///|
test "dynamic FP8 Program owns both quantization stages and cancels active work" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let (program, routed_bytes, shared_bytes) = dynamic_program_fixture()
  defer (program.abort() catch { _ => () })
  assert_eq(program.workspace_bytes(), 6454L)
  let external = [20L, 512L, 16L, 16L, routed_bytes, shared_bytes, 512L].map(bytes => {
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
    maximum_workspace_bytes=6454L,
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
      live_rows=2,
      plan_sequence=sequence,
      global_sequence=sequence,
    )
    while !program.poll() {
      ()
    }
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 176UL)
  program.submit(live_rows=1, plan_sequence=17UL, global_sequence=17UL)
  program.abort()
  program.close()
  for allocation in external.rev() {
    allocation.close()
  }
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

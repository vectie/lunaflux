///|
test "packed expert transport binds ten startup launches with bounded borrowed storage" {
  let before = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let module = context.load_module(b"fake\x00")
  defer (module.close() catch { _ => () })
  for transport in [@matrix.PackedRows, PackedPipeline] {
    let weight = @precision.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=NvFp4,
        rows=128,
        columns=128,
      ),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=1048576L,
      tile_elements=32,
    )
    let q = [3, 6].map(rows => {
      @precision.plan_operand(
        role=Activation,
        representation=@format.Representation::new(
          format=NvFp4,
          rows~,
          columns=128,
        ),
        placement=QuantizeBeforeConsumer,
        supported_placements=[QuantizeBeforeConsumer],
        accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
        memory_ceiling=1048576L,
      ).quantization()
    })
    let plan = @precision.ExpertMlpPrecision::new(
      gate=weight,
      up=weight,
      down=weight,
      expert_count=2,
      local_experts=[0, 1],
      selected_experts=2,
      maximum_rows=3,
      dot_law=MatrixFloat32,
      scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      activation_quantization=(q[0], q[1]),
      down_publication=ReduceFloat32BeforeRound,
      memory_ceiling=1048576L,
    )
    let source = @source.packed_expert_mlp_source(
      plan,
      symbol="packed_test",
      grouping=@matrix.ExpertGroupingPlan::new(
        maximum_rows=3,
        selected_experts=2,
        transport~,
      ),
    )
    assert_eq(source.symbols().length(), 10)
    assert_true(source.has_packed_rows())
    assert_eq(
      source.launches()[4].1,
      if transport == PackedPipeline {
        256
      } else {
        128
      },
    )
    assert_eq(source.launches()[8].1, source.launches()[4].1)
    let functions = source
      .symbols()
      .map(name => module.load_function(@utf8.encode(name)))
    defer (for f in functions.rev() {
      f.close() catch {
        _ => ()
      }
    })
    let qbytes = plan.quantized_workspace_bytes()
    let allocations = [
      20L,
      768L,
      24L,
      24L,
      8L,
      plan.bank_bytes(),
      source.activation_bytes(),
      plan.workspace_bytes().1,
      1536L,
      qbytes.0,
      qbytes.1,
      4L,
    ].map(n => context.allocate(n))
    defer (for a in allocations.rev() {
      a.close() catch {
        _ => ()
      }
    })
    let launches = @moe.bind(
      source,
      functions[:],
      counts=allocations[0],
      input=allocations[1],
      selected_indices=allocations[2],
      selected_weights=allocations[3],
      expert_map=allocations[4],
      bank=allocations[5],
      activation=allocations[6],
      weighted=allocations[7],
      output=allocations[8],
      quantized_input=allocations[9],
      quantized_activation=allocations[10],
      quantization_error=allocations[11],
    )
    assert_eq(launches.length(), 10)
    let queue = context.create_ordered_kernel_executor(stream, launches)
    defer (queue.close() catch { _ => () })
    begin_allocations()
    for _ in 0..<16 {
      for i in 0..<10 {
        queue.enqueue(i)
      }
      queue.record_completion()
      while !queue.poll_completion() {
        ()
      }
      queue.reset()
    }
    assert_eq(end_allocations(), 0UL)
    queue.close()
    for a in allocations.rev() {
      a.close()
    }
    for f in functions.rev() {
      f.close()
    }
  }
  module.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), before)
}

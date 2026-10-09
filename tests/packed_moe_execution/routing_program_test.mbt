///|
fn routed_program_fixture(
  token_hash? : Bool = false,
  complete_local? : Bool = false,
) -> (@moe.MoeProgram, Int64, Int64) raise {
  let router = @precision.MoeRoutingPrecision::new(
    weight=@precision.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=Bf16,
        rows=4,
        columns=32,
      ),
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
      memory_ceiling=256L,
    ),
    maximum_rows=2,
    group_count=2,
    group_score_topk=1,
    selected_group_count=1,
    selected_experts=2,
    score=Sigmoid,
    selection=if token_hash {
      TokenHash(vocabulary_size=3)
    } else {
      GroupedTopK
    },
    normalize=true,
    scaling=2.5,
    maximum_workspace_bytes=128L,
  )
  program_fixture(
    routing=@router.routing_source(router, symbol="router"),
    complete_local~,
  )
}

///|
test "router is queued before experts without token-step heap or extra completion" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let (program, routed_bytes, shared_bytes) = routed_program_fixture()
  defer (program.abort() catch { _ => () })
  assert_eq(program.workspace_bytes(), 1444L)
  let external = [20L, 128L, 256L, 16L, routed_bytes, shared_bytes, 128L].map(bytes => {
    context.allocate(bytes)
  })
  defer (for allocation in external.rev() {
    allocation.close() catch {
      _ => ()
    }
  })
  program.prepare_routed(
    context,
    aot,
    stream,
    maximum_workspace_bytes=1444L,
    counts=external[0],
    input=external[1],
    router_weight=external[2],
    correction_bias=external[3],
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
  assert_eq(enqueues() - before, 160UL)
  program.submit(live_rows=1, plan_sequence=17UL, global_sequence=17UL)
  program.abort()
  for allocation in external.rev() {
    allocation.close()
  }
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "router partial startup ownership can be released" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let buffer = context.allocate(20L)
  defer (buffer.close() catch { _ => () })
  let baseline = live_resources()
  let (program, _, _) = routed_program_fixture()
  defer (program.abort() catch { _ => () })
  try
    program.prepare_routed(
      context,
      aot,
      stream,
      maximum_workspace_bytes=1444L,
      counts=buffer,
      input=buffer,
      router_weight=buffer,
      correction_bias=buffer,
      routed_bank=buffer,
      shared_bank=buffer,
      output=buffer,
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    error => raise error
  } noraise {
    _ => abort("undersized input must not bind")
  }
  assert_true(live_resources() > baseline)
  program.close()
  assert_eq(live_resources(), baseline)
}

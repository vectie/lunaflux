///|
test "partitioned frontier still collapses equivalent complete schedules" {
  let request = functional_compiler_request([], query_rows=64)
  let frontier = compile_partitioned_attention_tile_frontier(request, [
    24, 32, 64,
  ])
  assert_eq(frontier.variants().length(), 2)
  assert_eq(frontier.variants()[0].parallel_workgroup_target(), 24)
  assert_eq(frontier.variants()[1].parallel_workgroup_target(), 64)
  assert_eq(
    frontier.variants()[0].compilation().digest(),
    compile_partitioned_attention_tile(request).digest(),
  )
}

///|
test "equal partition counts do not identify different tile schedules" {
  let base = functional_compiler_request([], query_rows=64)
  let observations : Array[AttentionRegisterObservation] = []
  for target in [32, 64] {
    let request = {
      ..base,
      capabilities: base.capabilities.with_parallel_workgroup_target(target),
    }
    let plans = @luna_attention_strategy.compile_attention_tile_candidate_plans(
      request.problem,
      request.capabilities,
    )
    for plan in plans {
      let compiled = schedule_partitioned_attention_tile(
        optimize_attention_tile(elaborate_attention_tile({ request, plan, })),
      ) catch {
        InvalidAttentionTileCompilation(ScheduleDerivation) => continue
        error => raise error
      }
      let allowed = compiled.schedule().key_value_partitions() == 2 &&
        (
          (target == 32 && plan.candidate().stable_id() == 314) ||
          (target == 64 && plan.candidate().stable_id() == 312)
        )
      observations.push(
        AttentionRegisterObservation::new(
          compiled.schedule().digest(),
          registers_per_thread=if allowed { 32 } else { 1024 },
        ),
      )
    }
  }
  let request = base.with_resource_feedback(
    AttentionResourceBudget::new(
      units=36,
      threads_per_unit=1024,
      registers_per_unit=65536,
      register_allocation_grain=256,
      shared_bytes_per_unit=100000L,
      blocks_per_unit=16,
    ),
    observations,
  )
  let frontier = compile_partitioned_attention_tile_frontier(request, [32, 64])
  assert_eq(frontier.variants().length(), 2)
  assert_eq(
    frontier.variants()[0].compilation().plan().candidate().stable_id(),
    314,
  )
  assert_eq(
    frontier.variants()[1].compilation().plan().candidate().stable_id(),
    312,
  )
  for variant in frontier.variants() {
    assert_eq(variant.compilation().schedule().key_value_partitions(), 2)
  }
}

///|
test "partitioned frontier retains every distinct independently selected schedule" {
  let targets = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
  for units in [1, 2, 36] {
    let request = functional_compiler_request([], query_rows=64).with_resource_feedback(
      AttentionResourceBudget::new(
        units~,
        threads_per_unit=1024,
        registers_per_unit=65536,
        register_allocation_grain=256,
        shared_bytes_per_unit=100000L,
        blocks_per_unit=16,
      ),
      [],
    )
    let frontier = compile_partitioned_attention_tile_frontier(request, targets)
    for target in targets {
      let compilation = compile_partitioned_attention_tile({
        ..request,
        capabilities: request.capabilities.with_parallel_workgroup_target(
          target,
        ),
      })
      assert_true(
        frontier
        .variants()
        .any(variant => variant.compilation().digest() == compilation.digest()),
      )
    }
  }
}

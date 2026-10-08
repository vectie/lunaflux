///|
test "joint enumeration retains all candidate target combinations before selection" {
  let request = functional_compiler_request([], query_rows=64)
  let joint = enumerate_partitioned_attention_tiles(request, [312, 314], [
    32, 64, 128,
  ])
  for target in [32, 64, 128] {
    for candidate in [312, 314] {
      let expected = specialize_partitioned_attention_tile(
        {
          ..request,
          capabilities: request.capabilities.with_parallel_workgroup_target(
            target,
          ),
        },
        candidate,
      ) catch {
        InvalidAttentionTileCompilation(ScheduleDerivation) => continue
        error => raise error
      }
      assert_true(
        joint.variants().any(v => v.compilation().digest() == expected.digest()),
      )
    }
  }
  assert_true(
    joint.variants().length() >
    compile_partitioned_attention_tile_frontier(request, [32, 64, 128])
    .variants()
    .length(),
  )
  assert_eq(
    joint,
    enumerate_partitioned_attention_tiles(request, [314, 312], [128, 32, 64]),
  )
}

///|
test "single kernel timings cannot prune complete chain alternatives" {
  let base = functional_compiler_request([], query_rows=64)
  let plans = @luna_attention_strategy.compile_attention_tile_candidate_plans(
    base.problem,
    base.capabilities,
  )
  guard plans.filter(p => p.candidate().stable_id() == 314) is [plan] else {
    fail("missing candidate")
  }
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend=base.backend,
      target=base.target,
      problem=base.problem,
      candidate=plan.candidate(),
    ),
    median_latency_ns=1L,
    sample_count=7,
  )
  assert_eq(
    enumerate_partitioned_attention_tiles(
      { ..base, autotune_records: [record], },
      [312, 314],
      [32, 64, 128],
    ),
    enumerate_partitioned_attention_tiles(base, [312, 314], [32, 64, 128]),
  )
}

///|
test "joint enumeration removes infeasible schedules not slower estimates" {
  let base = functional_compiler_request([], query_rows=64)
  let all = enumerate_partitioned_attention_tiles(base, [312, 314], [64, 128])
  let doomed = all.variants()[0].compilation().schedule().digest()
  let constrained = base.with_resource_feedback(
    AttentionResourceBudget::new(
      units=36,
      threads_per_unit=1024,
      registers_per_unit=65536,
      register_allocation_grain=256,
      shared_bytes_per_unit=100000L,
      blocks_per_unit=16,
    ),
    [AttentionRegisterObservation::new(doomed, registers_per_thread=1024)],
  )
  let retained = enumerate_partitioned_attention_tiles(
    constrained,
    [312, 314],
    [64, 128],
  )
  assert_false(
    retained.variants().any(v => v.compilation().schedule().digest() == doomed),
  )
  for variant in all.variants() {
    if variant.compilation().schedule().digest() != doomed {
      assert_true(
        retained
        .variants()
        .any(v => v.compilation().digest() == variant.compilation().digest()),
      )
    }
  }
}

///|
test "joint enumeration rejects unknown candidates duplicates and empty search" {
  let request = functional_compiler_request([], query_rows=64)
  for
    (ids, targets) in [
      ([99999], [32]),
      ([312, 312], [32]),
      ([312], [32, 32]),
      ([], [32]),
      ([312], []),
    ] {
    try enumerate_partitioned_attention_tiles(request, ids, targets) catch {
      InvalidAttentionTileCompilation(_) => ()
    } noraise {
      _ => fail("invalid joint domain accepted")
    }
  }
}

///|
fn check_equal_partition_aot_symbols(
  request : CudaPartitionedAttentionTileAotRequest,
) -> Unit raise {
  let observations : Array[
    @luna_attention_tile_compiler.AttentionRegisterObservation,
  ] = []
  for target in [32, 64] {
    let variant_request = {
      ..request,
      capabilities: request.capabilities.with_parallel_workgroup_target(target),
    }
    let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
      request.problem,
      variant_request.capabilities,
    )
    for candidate in candidates {
      let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
        key=@luna_attention_strategy.AttentionAutotuneKey::new(
          backend=request.backend,
          target=request.target_id,
          problem=request.problem,
          candidate~,
        ),
        median_latency_ns=100L,
        sample_count=7,
      )
      let compiled = @luna_attention_tile_compiler.compile_partitioned_attention_tile(
        variant_request.functional_request([record]),
      ) catch {
        @luna_attention_tile_compiler.AttentionTileCompilerError::InvalidAttentionTileCompilation(
          ScheduleDerivation
        ) => continue
        error => raise error
      }
      let allowed = compiled.schedule().key_value_partitions() == 2 &&
        (
          (target == 32 && candidate.stable_id() == 314) ||
          (target == 64 && candidate.stable_id() == 312)
        )
      observations.push(
        @luna_attention_tile_compiler.AttentionRegisterObservation::new(
          compiled.schedule().digest(),
          registers_per_thread=if allowed { 32 } else { 1024 },
        ),
      )
    }
  }
  let frontier = compile_cuda_partitioned_attention_tile_aot_frontier(
    request.with_resource_feedback(
      @luna_attention_tile_compiler.AttentionResourceBudget::new(
        units=36,
        threads_per_unit=1024,
        registers_per_unit=65536,
        register_allocation_grain=256,
        shared_bytes_per_unit=100000L,
        blocks_per_unit=16,
      ),
      observations,
    ),
    [],
    [32, 64],
  )
  assert_eq(frontier.variants().length(), 2)
  let first = frontier.variants()[0].compilation()
  let second = frontier.variants()[1].compilation()
  assert_eq(
    first.source().symbols(),
    (request.partial_symbol, request.merge_symbol),
  )
  assert_eq(
    second.source().symbols(),
    (request.partial_symbol + "_p2_v1", request.merge_symbol + "_p2_v1"),
  )
  assert_false(first.functional().digest() == second.functional().digest())
  assert_eq(first.functional().schedule().key_value_partitions(), 2)
  assert_eq(second.functional().schedule().key_value_partitions(), 2)
}

///|
test "partitioned symbols distinguish equal-count schedules" {
  let counts = [2, 2, 4, 2, 4, 8]
  let suffixes : Array[String] = []
  for index, count in counts {
    let suffix = partitioned_attention_symbol_suffix(
      index,
      count,
      counts[:index].any(previous => previous == count),
    )
    assert_false(suffixes.contains(suffix))
    suffixes.push(suffix)
  }
  assert_eq(suffixes, ["", "_p2_v1", "_p4", "_p2_v3", "_p4_v4", "_p8"])
}

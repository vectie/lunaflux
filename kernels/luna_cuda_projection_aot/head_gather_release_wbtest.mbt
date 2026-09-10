///|
test "reusable segmented greedy release lowers real selected head variants through register gather" {
  let parameters : ProjectionParameters = {
    family: LanguageModelHead,
    input_width: 1024,
    output_width: 151936,
    query_width: 0,
    intermediate_width: 0,
  }
  let operation = projection_fixture_operation(
    10,
    LanguageModelHead,
    [Activation(@plan.ActivationRef::new(9)), Tensor(@plan.TensorRef::new(0))],
    ProjectionShape(input_width=1024, output_width=151936),
    @plan.KernelCapabilityId::dense_projection(),
    runtime_inputs=[StepCounts, QueryRowOffsets],
  )
  let profile = row_variant_test_profile()
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(RuntimeInput(QueryRowOffsets), 132L, 4L),
    projection_fixture_operand(
      ActivationInput(@plan.ActivationRef::new(9)),
      1024L * 1024L * 2L,
      16L,
    ),
    projection_fixture_operand(
      WeightInput(@plan.TensorRef::new(0)),
      151936L * 1024L * 2L,
      16L,
    ),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(10)),
      1024L * 151936L * 2L,
      16L,
    ),
  ]
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let records = []
  for bound in [8, 256] {
    let selected = for
      candidate in baseline
      .dispatch_plan(bound)
      .prefill()
      .output_distribution_candidates() {
      if candidate.output_tiles_per_workgroup() == 1 {
        break candidate
      }
    } nobreak {
      fail("missing single-output-tile candidate")
    }
    records.push(row_variant_test_record(parameters, selected))
  }
  let runtime = projection_build_runtime_dispatch(
    parameters, 1024, target, records,
  )
  let lowered = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation,
    profile,
    target,
    projection_fixture_compiler(),
    operands,
    runtime,
    projection_fixture_limits(max_source_bytes=131072),
  )
  let symbol = "lunaflux_luna_language_model_head_bf16_release_v1_segmented_greedy_v2"
  assert_eq(lowered.function_symbol(), symbol)
  assert_eq(lowered.row_variants.length(), 2)
  let source = @utf8.decode(lowered.source_bytes())
  assert_true(source.contains("void lunaflux_greedy_bf16_partial_v2("))
  assert_true(source.contains("void lunaflux_greedy_bf16_merge_v2("))
  for variant in lowered.row_variants {
    assert_eq(variant.dimensions().shared_memory_bytes(), 0)
    assert_eq(variant.dimensions().block_x(), 32)
    assert_eq(variant.dimensions().grid_x(), 9496)
    let parts = source.split("void \{variant.function_symbol()}(").to_array()
    assert_eq(parts.length(), 2)
    let body = parts[1].split("#include").to_array()[0]
    assert_true(body.contains("input_low + k"))
    assert_true(body.contains("mma.sync.aligned.m16n8k16"))
    assert_false(body.contains("__shared__"))
    assert_false(body.contains("__syncthreads"))
  }
}

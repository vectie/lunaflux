///|
test "manifest pointwise families lower deterministically with exact ABIs" {
  let profile = pointwise_test_profile()
  let cases = [
    (
      pointwise_embedding_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ExternalInput(TokenIds), 16L, alignment=4L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(0)), 256L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(0)), 64L),
      ],
      @luna_cuda_pointwise_aot.EmbeddingLookup,
    ),
    (
      pointwise_rms_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(0)), 64L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(1)), 16L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(1)), 64L),
      ],
      RmsNorm,
    ),
    (
      pointwise_residual_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(0)), 64L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(1)), 64L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(2)), 64L),
      ],
      ResidualAdd,
    ),
    (
      pointwise_qk_rms_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(1)), 128L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(2)), 8L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(3)), 8L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(2)), 128L),
      ],
      QkRmsNorm,
    ),
    (
      pointwise_rotary_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(RuntimeInput(QueryPositions), 16L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(2)), 128L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(3)), 128L),
      ],
      PositionedRotary,
    ),
  ]
  for index, case in cases {
    let (operation, operands, family) = case
    let first = @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
      operation,
      profile,
      pointwise_test_entry(index + 1),
      pointwise_test_target(),
      pointwise_test_compiler(),
      operands,
      pointwise_test_limits(),
    )
    let second = @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
      operation,
      profile,
      pointwise_test_entry(index + 1),
      pointwise_test_target(),
      pointwise_test_compiler(),
      operands,
      pointwise_test_limits(),
    )
    assert_eq(first, second)
    assert_eq(first.family(), family)
    assert_false(first.manifest_bindable())
    assert_true(first.function_symbol().contains("_op_"))
    let source = @utf8.decode(first.source_bytes())
    match family {
      EmbeddingLookup => {
        assert_true(source.contains("const int32_t *token_ids"))
        assert_true(source.contains("token < 16"))
      }
      RmsNorm => {
        assert_true(source.contains("__shared__ float partial[256]"))
        assert_true(source.contains("0.00001f"))
      }
      QkRmsNorm => {
        assert_eq(first.launch_dimensions().grid_x(), 16)
        assert_true(source.contains("const __nv_bfloat16 *q_norm"))
        assert_true(source.contains("const __nv_bfloat16 *k_norm"))
        assert_true(source.contains("partial[0] / 4.0f + 0.000001f"))
        assert_true(source.contains("packed_head < 2 ? q_norm : k_norm"))
        assert_true(source.contains("output[base + lane] = input[base + lane]"))
      }
      ResidualAdd => assert_true(source.contains("__float2bfloat16_rn(sum)"))
      PositionedRotary => {
        assert_true(source.contains("powf(10000.0f, exponent)"))
        assert_true(source.contains("column >= 12"))
      }
      _ => fail("unexpected manifest family")
    }
    let recipe = @utf8.decode(first.recipe_bytes())
    assert_true(
      recipe.contains("schema=lunaflux-luna-cuda-pointwise-aot-recipe-v1"),
    )
    assert_true(recipe.contains("manifest_bindable=false"))
    assert_true(recipe.contains("reassociate=false"))
  }
}

///|
test "standalone KV write preserves canonical order but stays auxiliary" {
  let operation = pointwise_attention_operation()
  let layout = pointwise_test_layout()
  let layer = @plan.KvLayerId::new(0)
  let operands = [
    pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
    pointwise_operand(RuntimeInput(QueryPositions), 16L, alignment=4L),
    pointwise_operand(RuntimeInput(QueryRowOffsets), 12L, alignment=4L),
    pointwise_operand(RuntimeInput(SequenceLengths), 8L, alignment=4L),
    pointwise_operand(RuntimeInput(PageTableOffsets), 12L, alignment=4L),
    pointwise_operand(RuntimeInput(PageTablePageIndices), 16L, alignment=4L),
    pointwise_operand(ActivationInput(@plan.ActivationRef::new(3)), 128L),
    pointwise_operand(PersistentKvComponent(layer, Key), 768L),
    pointwise_operand(PersistentKvComponent(layer, Value), 768L),
  ]
  let lowering = @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
    operation,
    pointwise_test_profile(),
    layout,
    pointwise_test_entry(9),
    pointwise_test_target(),
    pointwise_test_compiler(),
    operands,
    pointwise_test_limits(),
  )
  assert_eq(lowering.family(), KvCacheWrite)
  assert_false(lowering.manifest_bindable())
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("[page][token][kv-head][head-dim]") == false)
  assert_true(source.contains("physical_page * 128ULL"))
  assert_true(source.contains("key_cache[cache_index]"))
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(recipe.contains("manifest_bindable=false"))
  assert_true(recipe.contains("family=kv_cache_write"))
}

///|
test "strict manifest join binds exact contract family and module" {
  let plan = pointwise_manifest_plan()
  let operation = plan.operations()[0]
  let operands = pointwise_manifest_operands(0)
  let candidate = @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
    operation,
    pointwise_test_profile(),
    pointwise_manifest_entry(0).entry_point_id(),
    pointwise_test_target(),
    pointwise_test_compiler(),
    operands,
    pointwise_test_limits(),
  )
  let bound = @luna_cuda_pointwise_aot.bind_manifest_pointwise_cuda_aot(
    candidate,
    operation,
    pointwise_manifest_contracts(),
  )
  assert_true(bound.manifest_bindable())
  assert_eq(bound.candidate(), candidate)
  assert_eq(bound.entry_point(), pointwise_manifest_entry(0))
  let record = @utf8.decode(bound.binding_record_bytes())
  assert_true(record.contains("manifest_bindable=true"))
  assert_true(record.contains("module_sha256=6666666666666666"))
  assert_true(record.contains("operands=runtime:step_counts@20/4"))
}

///|
test "strict manifest join rejects contract dimension drift" {
  let plan = pointwise_manifest_plan()
  let operation = plan.operations()[0]
  let candidate = @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
    operation,
    pointwise_test_profile(),
    pointwise_manifest_entry(0).entry_point_id(),
    pointwise_test_target(),
    pointwise_test_compiler(),
    pointwise_manifest_operands(0),
    pointwise_test_limits(),
  )
  try
    @luna_cuda_pointwise_aot.bind_manifest_pointwise_cuda_aot(
      candidate,
      operation,
      pointwise_manifest_contracts(embedding_grid=2),
    )
  catch {
    PointwiseAotError::Mismatch(_) => ()
    _ => fail("unexpected manifest rejection")
  } noraise {
    _ => fail("expected dimension mismatch")
  }
}

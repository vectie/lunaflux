///|
test "QKNorm RoPE positioned KV-write candidate is deterministic and postprocess-only" {
  let fixture = postprocess_test_fixture()
  let first = postprocess_test_candidate(fixture)
  let second = postprocess_test_candidate(fixture)
  assert_eq(first.source_digest(), second.source_digest())
  assert_eq(first.recipe_digest(), second.recipe_digest())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(
    first.function_symbol(),
    "lunaflux_qwen_qknorm_rope_kvwrite_bf16_head128_production_v2",
  )
  assert_eq(
    first.kernel_abi(),
    QwenQkNormPositionedRopePagedKvWriteProductionV2,
  )
  assert_eq(
    first.kernel_abi().canonical_name(),
    "step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,projected_qkv," +
    "query_norm_weight,key_norm_weight,rotated_qkv,key_cache,value_cache",
  )
  assert_eq(first.launch_dimensions().grid_x(), 4096)
  assert_eq(first.launch_dimensions().grid_y(), 8)
  assert_eq(first.launch_dimensions().block_x(), 128)
  assert_false(first.manifest_bindable())
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("#define LF_HEAD_DIM 128\n"))
  assert_true(
    source.contains("const __nv_bfloat16* __restrict__ projected_qkv"),
  )
  assert_true(
    source.contains("const __nv_bfloat16* __restrict__ query_norm_weight"),
  )
  assert_true(
    source.contains("const __nv_bfloat16* __restrict__ key_norm_weight"),
  )
  assert_true(source.contains("token += (int)gridDim.x"))
  assert_true(source.contains("gridDim.x == 0 || gridDim.x > 4096"))
  assert_false(source.contains("for (int row_check"))
  assert_true(source.contains("lane < LF_HEAD_DIM / 2"))
  assert_true(source.contains("rope_cosine[lane] = cosf(angle)"))
  assert_true(source.contains("rotated_qkv[head_base + component] = encoded"))
  assert_true(
    source.contains("(key ? key_cache : value_cache)[cache_index] = encoded"),
  )
  assert_false(source.contains("q_weight"))
  assert_false(source.contains("k_weight"))
  assert_false(source.contains("v_weight"))
  assert_false(source.contains("LF_INPUT_WIDTH"))
  assert_false(source.contains("nvcuda::wmma"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(recipe.contains("family=qknorm-positioned-rope-paged-kv-write\n"))
  assert_true(recipe.contains("projection_included=false\n"))
  assert_true(recipe.contains("projection_weights_read=false\n"))
  assert_true(recipe.contains("gemm_included=false\n"))
  assert_true(recipe.contains("grid_stride_tokens=true\n"))
  assert_true(recipe.contains("grid_x_policy=query-token-capped\n"))
  assert_true(recipe.contains("manifest_bindable=false\n"))
}

///|
test "QKNorm RoPE positioned KV-write ABI requires disjoint writable regions" {
  let contract = postprocess_test_candidate(postprocess_test_fixture()).alias_contract()
  assert_eq(
    contract.canonical_name(),
    "readonly[0,9);writable[9,12);writable-disjoint-from-all",
  )
  assert_false(contract.requires_nonoverlap(0, 8))
  assert_true(contract.requires_nonoverlap(0, 9))
  assert_true(contract.requires_nonoverlap(9, 10))
  assert_true(contract.requires_nonoverlap(10, 11))
  assert_false(contract.requires_nonoverlap(9, 9))
  assert_false(contract.requires_nonoverlap(-1, 9))
}

///|
test "QKNorm RoPE positioned KV-write rejects operand size substitution" {
  let fixture = postprocess_test_fixture()
  let changed = fixture.operands.copy()
  changed[9] = postprocess_test_operand(
    changed[9].role(),
    changed[9].byte_count() - 2L,
    changed[9].alignment(),
  )
  try postprocess_test_candidate(fixture, operands=changed[:]) catch {
    Mismatch(OperandSize) => ()
    _ => fail("unexpected postprocess operand rejection")
  } noraise {
    _ => fail("substituted postprocess operand reached the candidate")
  }
}

///|
test "head128 production symbol rejects other head geometry" {
  let fixture = postprocess_test_fixture(head_dimension=32)
  try postprocess_test_candidate(fixture) catch {
    Unsupported(Shape) => ()
    _ => fail("unexpected non-head128 geometry rejection")
  } noraise {
    _ => fail("non-head128 geometry reached head128 production symbol")
  }
}

///|
test "compiled QKNorm RoPE positioned KV-write binds reproducible bytes only" {
  let candidate = postprocess_test_candidate(postprocess_test_fixture())
  let module_ = b"fixture-qknorm-rope-positioned-kv-write-cubin"
  let receipt = qknorm_rope_paged_kv_write_compile_receipt(
    candidate,
    "d".repeat(64),
    module_,
  )
  let compiled = admit_reproducibly_compiled_qknorm_rope_paged_kv_write_module(
    candidate,
    module_,
    module_,
    receipt,
    postprocess_test_limits(),
  )
  let binding = bind_compiled_qknorm_rope_paged_kv_write_cuda_aot(
    candidate,
    compiled,
    postprocess_test_limits(),
  )
  assert_eq(binding.function_symbol(), candidate.function_symbol())
  assert_eq(binding.kernel_abi(), candidate.kernel_abi())
  assert_eq(binding.module_digest(), compiled.module_digest())
  assert_false(binding.manifest_bindable())
  let record = @utf8.decode(binding.binding_record_bytes())
  assert_true(record.contains("operation_span=3,4,5\n"))
  assert_true(record.contains("manifest_bindable=false\n"))
  try
    admit_reproducibly_compiled_qknorm_rope_paged_kv_write_module(
      candidate,
      module_,
      b"different-module",
      receipt,
      postprocess_test_limits(),
    )
  catch {
    Mismatch(Determinism) => ()
    _ => fail("unexpected non-deterministic module rejection")
  } noraise {
    _ => fail("non-deterministic module reached postprocess binding")
  }
}

///|
test "paged attention lowering is deterministic and preserves exact v3 ABI" {
  let first = pa_lowering()
  let second = pa_lowering()
  assert_eq(first, second)
  assert_eq(first.query_heads(), 2)
  assert_eq(first.kv_heads(), 1)
  assert_eq(first.head_dimension(), 2)
  assert_eq(first.tokens_per_page(), 2)
  assert_eq(first.page_stride_bytes(), 256L)
  assert_eq(first.physical_page_capacity(), 4L)
  assert_eq(first.launch_dimensions().grid_x(), 3)
  assert_eq(first.launch_dimensions().grid_y(), 2)
  assert_eq(first.launch_dimensions().block_x(), 32)
  assert_eq(first.launch_dimensions().shared_memory_bytes(), 0)
  assert_eq(first.operands().length(), 10)
  assert_eq(first.entry_point_id(), pa_entry(3).entry_point_id())
  assert_eq(
    first.source_digest().as_hex(),
    "93883127bd5039aaf8810b4eeaef89e7f7bc13e46df29b5ca0613cb7d0bb36e0",
  )
  assert_eq(
    first.recipe_digest().as_hex(),
    "b373b71fb2b38b2a935d95d853075d988de377559cb9aa4ba2d5c044e6bc7d20",
  )
  assert_eq(first.operands()[0].role(), RuntimeInput(StepCounts))
  assert_eq(
    first.operands()[6].role(),
    ActivationInput(@plan.ActivationRef::new(1)),
  )
  assert_eq(
    first.operands()[9].role(),
    PersistentKvComponent(@plan.KvLayerId::new(0), Value),
  )
}

///|
test "generated source covers mixed prefill decode without grid synchronization" {
  let lowering = pa_lowering()
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("const int prefill_count = counts[0]"))
  assert_true(source.contains("decode_count = counts[1]"))
  assert_true(source.contains("key_cache[cache_index] = qkv["))
  assert_true(source.contains("in_chunk") == false)
  assert_true(source.contains("key_position >= chunk_start"))
  assert_true(source.contains("maximum = -CUDART_INF_F"))
  assert_true(source.contains("accumulator[item] / denominator"))
  assert_true(source.contains("__shfl_down_sync"))
  assert_false(source.contains("__syncthreads"))
  assert_false(source.contains("malloc"))
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(recipe.contains("mode=mixed-prefill-decode\n"))
  assert_true(recipe.contains("language_standard=c++17\n"))
  assert_true(recipe.contains("grid=3,2,1\n"))
  assert_true(recipe.contains("block=32,1,1\n"))
  assert_true(recipe.contains("shared_memory_bytes=0\n"))
  assert_true(recipe.contains("manifest_bindable=false\n"))
  assert_false(recipe.contains("module_sha256="))
  assert_true(recipe.contains("kv_write=fused-current-token\n"))
  assert_true(recipe.contains("in_chunk_kv_source=qkv-activation\n"))
  assert_true(recipe.contains("fmad=false\n"))
}

///|
test "binder rejects foreign geometry and lowering rejects target/compiler semantics" {
  let candidate = pa_lowering()
  try
    @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_cuda_aot(
      candidate,
      pa_plan().operations()[2],
      pa_contracts(exact_dimensions=false),
    )
  catch {
    @luna_cuda_paged_attention_aot.Mismatch(Contract) => ()
    _ => fail("unexpected dimension rejection")
  } noraise {
    _ => fail("expected exact dimension rejection")
  }
  try {
    let contract = pa_attention_contract()
    @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
      pa_plan().operations()[2],
      contract.profile(),
      pa_layout(),
      contract.entry_point().entry_point_id(),
      pa_target(bf16=false),
      pa_compiler(),
      contract.operands(),
      pa_limits(),
    )
  } catch {
    @luna_cuda_paged_attention_aot.Unsupported(DeviceTarget) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("expected BF16 target rejection")
  }
  try {
    let contract = pa_attention_contract()
    @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
      pa_plan().operations()[2],
      contract.profile(),
      pa_layout(),
      contract.entry_point().entry_point_id(),
      pa_target(),
      pa_compiler(reassociate=true),
      contract.operands(),
      pa_limits(),
    )
  } catch {
    @luna_cuda_paged_attention_aot.Unsupported(CompilerPolicy) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("expected reassociation rejection")
  }
}

///|
test "strict binder authenticates final attention module after compilation" {
  let candidate = pa_lowering()
  let contracts = pa_contracts()
  let bound = @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_cuda_aot(
    candidate,
    pa_plan().operations()[2],
    contracts,
  )
  assert_true(bound.manifest_bindable())
  assert_eq(bound.candidate(), candidate)
  assert_eq(bound.entry_point(), pa_entry(3))
  let record = @utf8.decode(bound.binding_record_bytes())
  assert_true(record.contains("module_sha256=" + "3".repeat(64)))
  assert_true(record.contains("manifest_bindable=true\n"))
}

///|
test "digest constructors reject non-canonical values" {
  try
    @luna_cuda_paged_attention_aot.PagedAttentionCudaSourceDigest::from_sha256(
      "ABC",
    )
  catch {
    Invalid(Source) => ()
    _ => fail("unexpected source digest rejection")
  } noraise {
    _ => fail("expected source digest rejection")
  }
}

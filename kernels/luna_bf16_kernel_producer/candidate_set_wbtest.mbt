///|
fn tiny_candidate_plan() -> @plan.ModelPlan raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=3000,
    hidden_size=16,
    intermediate_size=64,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=4,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256(
        "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30",
      ),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
}

///|
fn tiny_candidate_layout(
  plan : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=8,
      total_page_count=1,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn tiny_candidate_target(minor? : Int = 0) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn tiny_candidate_profile(
  rows? : Int = 1,
) -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=rows,
    max_query_tokens=rows,
    max_page_table_entries=1,
  )
}

///|
fn tiny_candidate_limits(
  max_operations? : Int = 64,
) -> LunaBf16CandidateSetLimits raise {
  LunaBf16CandidateSetLimits::new(
    max_operations~,
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
    max_total_bytes=33554432,
  )
}

///|
fn tiny_candidate_set() -> LunaBf16CandidateSet raise {
  let plan = tiny_candidate_plan()
  prepare_luna_bf16_candidate_set(
    plan,
    tiny_candidate_layout(plan),
    tiny_candidate_profile(),
    tiny_candidate_target(),
    producer_fixture_compiler(),
    tiny_candidate_limits(),
  )
}

///|
test "authenticated tiny BF16 plan produces canonical sm120 candidate set" {
  let set = tiny_candidate_set()
  let identity = set.model_identity()
  assert_eq(
    identity.content().as_hex(),
    "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30",
  )
  assert_eq(
    identity.plan().as_hex(),
    "22103eddad1277154b79321e53893ad579e47d040d62b896f5596624e0b7be3e",
  )
  assert_eq(set.operation_count(), 21)
  assert_eq(set.candidates().length(), 21)
  assert_eq(set.candidate_key(0), Some("op_00000000_embedding_lookup"))
  assert_eq(set.candidate_key(20), Some("op_00000020_language_model_head"))
  assert_eq(set.candidate_key(21), None)
  let head_source = @utf8.decode(set.candidate_source_bytes(20).unwrap())
  assert_true(
    head_source.contains(
      "extern \"C\" __global__ void lunaflux_greedy_bf16_rows_v1(",
    ),
  )
  assert_true(head_source.contains("if (!isfinite(value))"))
  let head_recipe = @utf8.decode(set.candidate_recipe_bytes(20).unwrap())
  assert_true(head_recipe.contains("embedded_sampling=greedy_bf16_rows_v1\n"))
  assert_true(head_recipe.contains("sampling_maximum_rows=1\n"))
  assert_true(head_recipe.contains("sampling_vocabulary_size=3000\n"))
  let preceding_source = @utf8.decode(set.candidate_source_bytes(19).unwrap())
  assert_false(preceding_source.contains("lunaflux_greedy_bf16_rows_v1"))
  let declaration = @utf8.decode(set.candidate_set_bytes())
  assert_true(
    declaration.has_prefix(
      "schema=lunaflux-bf16-candidate-set.v1\ntarget=sm_120\noperation_count=21\n",
    ),
  )
  let inventory = @utf8.decode(set.inventory_bytes())
  let lines = inventory.split("\n").to_array()
  assert_eq(lines.length(), 44)
  assert_eq(lines[43], "")
  assert_true(
    lines[0].has_suffix("  candidate-set.v1") &&
    lines[1].contains("  recipes/op_00000000_embedding_lookup.recipe") &&
    lines[22].contains("  sources/op_00000000_embedding_lookup.cu"),
  )
  for index, candidate in set.candidates() {
    assert_eq(candidate.operation_id().as_int(), index)
    assert_eq(candidate.target(), set.target())
    assert_eq(
      candidate.source_sha256(),
      sha256_bytes(set.candidate_source_bytes(index).unwrap()),
    )
    assert_eq(
      candidate.recipe_sha256(),
      sha256_bytes(set.candidate_recipe_bytes(index).unwrap()),
    )
  }
}

///|
test "candidate sequence rejects reordered and cross-target lowerings" {
  let plan = tiny_candidate_plan()
  let set = tiny_candidate_set()
  let reordered : Array[LunaBf16KernelCandidate] = []
  for candidate in set.candidates() {
    reordered.push(candidate)
  }
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  try validate_candidate_sequence(plan, set.target(), reordered) catch {
    Mismatch(OperationIdentity) => ()
    _ => fail("unexpected reordered-candidate rejection")
  } noraise {
    _ => fail("expected reordered candidate rejection")
  }
  let foreign = prepare_luna_bf16_candidate_set(
    plan,
    tiny_candidate_layout(plan),
    tiny_candidate_profile(),
    tiny_candidate_target(minor=1),
    producer_fixture_compiler(),
    tiny_candidate_limits(),
  )
  let substituted : Array[LunaBf16KernelCandidate] = []
  for candidate in set.candidates() {
    substituted.push(candidate)
  }
  substituted[0] = foreign.candidates()[0]
  try validate_candidate_sequence(plan, set.target(), substituted) catch {
    Mismatch(DeviceTarget) => ()
    _ => fail("unexpected target-substitution rejection")
  } noraise {
    _ => fail("expected target-substitution rejection")
  }
}

///|
test "candidate set rejects layout profile and capacity substitution" {
  let plan = tiny_candidate_plan()
  let other = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      @spec.LlamaModelSpec::new(
        vocabulary_size=3000,
        hidden_size=16,
        intermediate_size=64,
        layer_count=2,
        attention_head_count=4,
        kv_head_count=4,
        context_length=256,
        rope_theta=10000.0,
        rms_norm_epsilon=0.00001,
        weight_dtype=BFloat16,
        kv_dtype=BFloat16,
      ),
      @spec.ContentDigest::from_sha256("4".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  try
    prepare_luna_bf16_candidate_set(
      plan,
      tiny_candidate_layout(other),
      tiny_candidate_profile(),
      tiny_candidate_target(),
      producer_fixture_compiler(),
      tiny_candidate_limits(),
    )
  catch {
    Mismatch(CandidateSet) => ()
    _ => fail("unexpected layout-substitution rejection")
  } noraise {
    _ => fail("expected layout-substitution rejection")
  }
  try
    prepare_luna_bf16_candidate_set(
      plan,
      tiny_candidate_layout(plan),
      tiny_candidate_profile(rows=2),
      tiny_candidate_target(),
      producer_fixture_compiler(),
      tiny_candidate_limits(),
    )
  catch {
    Mismatch(Profile) => ()
    _ => fail("unexpected profile-substitution rejection")
  } noraise {
    _ => fail("expected profile-substitution rejection")
  }
  try
    prepare_luna_bf16_candidate_set(
      plan,
      tiny_candidate_layout(plan),
      tiny_candidate_profile(),
      tiny_candidate_target(),
      producer_fixture_compiler(),
      tiny_candidate_limits(max_operations=20),
    )
  catch {
    LimitExceeded(OperationCoverage, maximum=20) => ()
    _ => fail("unexpected operation-ceiling rejection")
  } noraise {
    _ => fail("expected operation-ceiling rejection")
  }
}

///|
test "packed ingress exports head ownership and reuses row rotary before head loop" {
  let baseline = qwen_fused_test_candidate(
    head_dimension=64,
    maximum_tokens=33,
    query_heads=6,
    kv_heads=3,
  )
  for heads in [1, 2, 4] {
    let candidate = qwen_fused_test_candidate(
      head_dimension=64,
      maximum_tokens=33,
      query_heads=6,
      kv_heads=3,
      ingress_policy=qwen_ingress_policy(4, 2, 1, 1, heads~),
    )
    assert_eq(
      candidate.tile_compilation().program().digest(),
      baseline.tile_compilation().program().digest(),
    )
    assert_eq(candidate.launch_dimensions().grid_y(), (12 - 1) / heads + 1)
    let source = @utf8.decode(candidate.source_bytes())
    assert_false(source.contains("void lf_ingress_norm_rotary("))
    assert_true(source.contains("#define LF_HEADS_PER_CTA \{heads}\n"))
    assert_true(source.contains("#define LF_PROJECTION_WIDTH \{64 * heads}\n"))
    assert_true(
      source.contains("const bool valid = weight_head < LF_PACKED_HEADS;"),
    )
    assert_true(source.contains("local_column % LF_HEAD_DIM"))
    assert_true(
      source.contains(
        "if (head_begin < LF_QK_HEADS) rotary = lf_ingress_prepare_rotary(position, lane);\n    #pragma unroll\n    for (int local_head",
      ),
    )
    assert_true(
      source.contains(
        "lf_ingress_norm_rotary_prepared(head_values, norm_weight, rotary, lane)",
      ),
    )
    assert_true(source.contains("if (head >= LF_PACKED_HEADS) continue;"))
    let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      candidate,
      fused_test_limits(),
    )
    assert_true(
      @utf8.decode(production.recipe_bytes()).contains(
        "heads_per_cta=\{heads}\n",
      ),
    )
    assert_eq(production.ingress_policy().head_tiles(), heads)
  }
  // Three packed heads create a genuine masked tail for a two-head CTA.
  let tail = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=33,
    query_heads=1,
    kv_heads=1,
    ingress_policy=qwen_ingress_policy(4, 2, 1, 1, heads=2),
  )
  assert_eq(tail.launch_dimensions().grid_y(), 2)
}

///|
test "packed ingress resource frontier retains single head and rejects excess storage" {
  let candidate = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=33,
  )
  let policies = qwen_ingress_policies(candidate.tile_compilation())
  assert_true(policies.contains(candidate.ingress_policy()))
  assert_true(policies.contains(qwen_ingress_policy(4, 2, 1, 1, heads=2)))
  assert_true(policies.contains(qwen_ingress_policy(8, 2, 1, 1, heads=4)))
  assert_false(
    policies.contains(qwen_ingress_policy(8, 2, 1, 1, rows=2, heads=4)),
  )
  try
    qwen_fused_test_candidate(
      head_dimension=128,
      maximum_tokens=33,
      ingress_policy=qwen_ingress_policy(8, 2, 1, 1, rows=2, heads=4),
    )
  catch {
    FusedParallelAotError::FusedParallelUnsupported(StaticResources) => ()
    _ => fail("wrong packed resource rejection")
  } noraise {
    _ => fail("packed shared overflow accepted")
  }
}

///|
test "disjoint ingress scratch admits wider row and head geometry safely" {
  let candidate = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=65,
    ingress_policy=qwen_ingress_policy(4, 2, 2, 1, rows=2, heads=2),
  )
  let physical = qwen_refine_physical(
    candidate.tile_compilation(),
    candidate.ingress_policy(),
  )
  assert_true(physical.scratch.reuses_storage())
  assert_eq(physical.scratch.allocation_bytes(), 36864L)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("ingress_scratch[36864]"))
  assert_true(
    source.contains(
      "Retire every operand reader before reusing its storage.\n      __syncthreads();",
    ),
  )
  let overlapping = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=65,
    ingress_policy=qwen_ingress_policy(2, 2, 1, 1, heads=2),
  )
  let retained = qwen_refine_physical(
    overlapping.tile_compilation(),
    overlapping.ingress_policy(),
  )
  assert_false(retained.scratch.reuses_storage())
  assert_true(
    @utf8.decode(overlapping.source_bytes()).contains(
      "__shared__ float projected[",
    ),
  )
}

///|
fn parallel_test_plan(head_dimension? : Int = 32) -> @plan.ModelPlan raise {
  let query_heads = 4
  let hidden_size = query_heads * head_dimension
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=64,
    hidden_size~,
    intermediate_size=hidden_size * 2,
    layer_count=1,
    attention_head_count=query_heads,
    kv_head_count=2,
    context_length=128,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::new(2),
  )
}

///|
fn parallel_test_compiler() -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="b".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn parallel_reference_candidate(
  head_dimension? : Int = 32,
) -> @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering raise {
  let plan = parallel_test_plan(head_dimension~)
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=16,
      max_arena_bytes=16777216L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=8,
  )
  let set = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    @catalog.DeviceTarget::new(
      compute_major=9,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
    parallel_test_compiler(),
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
  for value in set.candidates() {
    match value {
      PagedAttention(candidate) => return candidate
      _ => ()
    }
  }
  abort("parallel attention fixture lacked reference candidate")
}

///|
fn parallel_test_limits() -> PagedAttentionParallelAotLimits raise {
  PagedAttentionParallelAotLimits::new(
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
    max_canonical_bytes=65536,
  )
}

///|
fn parallel_candidate() -> PagedAttentionParallelCudaCandidate raise {
  lower_paged_attention_parallel_cuda_aot_candidate(
    parallel_reference_candidate(),
    parallel_test_limits(),
  )
}

///|
test "parallel attention lowering is deterministic typed and genuinely parallel" {
  let first = parallel_candidate()
  let second = parallel_candidate()
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.recipe_bytes(), second.recipe_bytes())
  assert_eq(first.launch_dimensions().block_x(), 128)
  assert_eq(first.launch_dimensions().block_y(), 1)
  assert_eq(first.maximum_context_tokens(), 128)
  assert_eq(first.shared_memory_bytes(), (128 + 128) * 4)
  assert_eq(
    first.launch_dimensions().shared_memory_bytes(),
    first.shared_memory_bytes(),
  )
  assert_false(first.manifest_bindable())
  assert_true(
    first.reference_fallback().source_digest().as_hex() !=
    first.source_digest().as_hex(),
  )
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("threadIdx.x"))
  assert_true(source.contains("linear += LF_BLOCK_THREADS"))
  assert_true(source.contains("lf_parallel_reduce_sum_v1"))
  assert_true(source.contains("lf_parallel_reduce_max_v1"))
  assert_true(source.contains("__syncthreads"))
  assert_true(source.contains("extern __shared__ float shared[]"))
  assert_false(source.contains("blockIdx.z"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(recipe.contains("family=paged_attention_parallel_block128\n"))
  assert_true(recipe.contains("block=128,1,1\n"))
  assert_true(
    recipe.contains(
      "numerics=bf16-input-fp32-block128-tree-softmax-tolerance-v1\n",
    ),
  )
  assert_true(recipe.contains("dispatch_canary=parallel-block128-symbol-v1\n"))
  assert_true(
    recipe.contains("release_binding=candidate-only-promotion-required\n"),
  )
  assert_true(recipe.contains("manifest_bindable=false\n"))
}

///|
test "parallel attention shape and arithmetic boundaries fail closed" {
  try
    lower_paged_attention_parallel_cuda_aot_candidate(
      parallel_reference_candidate(head_dimension=16),
      parallel_test_limits(),
    )
  catch {
    PagedAttentionParallelUnsupported(Shape) => ()
    _ => fail("unexpected head-dimension rejection")
  } noraise {
    _ => fail("unsupported head dimension was lowered")
  }
  try parallel_checked_product(2147483647, 2) catch {
    PagedAttentionParallelOverflow(StaticResources) => ()
    _ => fail("unexpected static-resource overflow rejection")
  } noraise {
    _ => fail("static-resource overflow was admitted")
  }
  try
    PagedAttentionParallelAotLimits::new(
      max_source_bytes=0,
      max_recipe_bytes=1,
      max_canonical_bytes=1,
    )
  catch {
    PagedAttentionParallelInvalid(Limits) => ()
    _ => fail("unexpected candidate limit rejection")
  } noraise {
    _ => fail("invalid candidate limits were admitted")
  }
}

///|
fn parallel_host_attention(
  qkv : @reference.Activation,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
) -> (@reference.Activation, Int) raise {
  let rows = qkv.rows()
  let query_width = query_heads * head_dimension
  let kv_width = kv_heads * head_dimension
  let qkv_width = query_width + 2 * kv_width
  let group_size = query_heads / kv_heads
  let output = Array::make(rows * query_width, 0.0F)
  let scores = FixedArray::make(rows, 0.0F)
  let partials = FixedArray::make(128, 0.0F)
  let scale = 1.0F / Float::from_int(head_dimension).sqrt()
  let mut canary = 0
  for query_row in 0..<rows {
    for query_head in 0..<query_heads {
      canary += 1
      let kv_head = query_head / group_size
      let query_offset = query_row * qkv_width + query_head * head_dimension
      for key_row in 0..<=query_row {
        for lane in 0..<128 {
          let mut partial = 0.0F
          for component = lane
              component < head_dimension
              component = component + 128 {
            partial += qkv.values()[query_offset + component] *
              qkv.values()[key_row * qkv_width +
              query_width +
              kv_head * head_dimension +
              component]
          }
          partials[lane] = partial
        }
        for stride = 64; stride > 0; stride = stride / 2 {
          for lane in 0..<stride {
            partials[lane] += partials[lane + stride]
          }
        }
        scores[key_row] = partials[0] * scale
      }
      let mut maximum = scores[0]
      for key_row in 1..<=query_row {
        maximum = maximum.max(scores[key_row])
      }
      let mut denominator = 0.0F
      for key_row in 0..<=query_row {
        scores[key_row] = @math.expf(scores[key_row] - maximum)
        denominator += scores[key_row]
      }
      for component in 0..<head_dimension {
        let mut accumulator = 0.0F
        for key_row in 0..<=query_row {
          accumulator += scores[key_row] /
            denominator *
            qkv.values()[key_row * qkv_width +
            query_width +
            kv_width +
            kv_head * head_dimension +
            component]
        }
        output[query_row * query_width + query_head * head_dimension + component] = accumulator
      }
    }
  }
  (
    @reference.Activation::new(
      rows~,
      columns=query_width,
      values=output,
      limits=@reference.ReferenceLimits::new(
        max_rows=rows,
        max_columns=qkv_width,
        max_elements=rows * qkv_width,
      ),
    ),
    canary,
  )
}

///|
test "parallel tree canary agrees with independent scalar attention" {
  let rows = 3
  let query_heads = 4
  let kv_heads = 2
  let head_dimension = 32
  let columns = (query_heads + 2 * kv_heads) * head_dimension
  let values = Array::make(rows * columns, 0.0F)
  for index in 0..<values.length() {
    values[index] = Float::from_int(index % 17 - 8) * 0.03125F
  }
  let limits = @reference.ReferenceLimits::new(
    max_rows=rows,
    max_columns=columns,
    max_elements=rows * columns,
  )
  let qkv = @reference.Activation::new(rows~, columns~, values~, limits~)
  let expected = @reference.causal_gqa_attention(
    qkv,
    query_heads~,
    kv_heads~,
    head_dimension~,
    limits~,
  )
  let (actual, canary) = parallel_host_attention(
    qkv, query_heads, kv_heads, head_dimension,
  )
  assert_eq(canary, rows * query_heads)
  for index in 0..<actual.values().length() {
    let delta = actual.values()[index] - expected.values()[index]
    let absolute = if delta < 0.0F { -delta } else { delta }
    assert_true(absolute <= 0.004F)
  }
}

///|
fn parallel_digest(digit : Char) -> PagedAttentionParallelDigest raise {
  PagedAttentionParallelDigest::from_sha256(digit.to_string().repeat(64))
}

///|
fn parallel_promotion_trials(
  slow? : Bool = false,
) -> Array[PagedAttentionParallelPromotionTrial] raise {
  let result = []
  for shape in 0..<2 {
    for trial in 0..<3 {
      result.push(
        PagedAttentionParallelPromotionTrial::new(
          shape_ordinal=shape,
          trial_ordinal=trial,
          baseline_duration_ns=1000L,
          candidate_duration_ns=if slow && shape == 0 { 1100L } else { 900L },
          maximum_absolute_error_ppb=1000L,
          maximum_relative_error_ppb=1000L,
          token_agreement=true,
          correctness=parallel_digest('c'),
          dispatch_canary=parallel_digest('d'),
        ),
      )
    }
  }
  result
}

///|
fn parallel_end_to_end() -> Array[PagedAttentionParallelEndToEndTrial] raise {
  let result = []
  for trial in 0..<2 {
    result.push(
      PagedAttentionParallelEndToEndTrial::new(
        trial_ordinal=trial,
        baseline_duration_ns=10000L,
        candidate_duration_ns=10000L,
        token_agreement=true,
        correctness=parallel_digest('e'),
      ),
    )
  }
  result
}

///|
fn parallel_promotion_limits() -> PagedAttentionParallelPromotionLimits raise {
  PagedAttentionParallelPromotionLimits::new(
    maximum_shapes=8,
    trials_per_shape=3,
    end_to_end_trials=2,
    maximum_end_to_end_regression_ppm=0,
    maximum_canonical_bytes=1048576,
  )
}

///|
test "candidate-only release binding requires exact promotion evidence" {
  let candidate = parallel_candidate()
  let evidence = admit_paged_attention_parallel_promotion_evidence(
    candidate,
    parallel_promotion_trials(),
    parallel_end_to_end(),
    parallel_promotion_limits(),
  )
  assert_eq(evidence.trial_count(), 6)
  let entry = @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=@catalog.AotArtifactDigest::from_sha256("f".repeat(64)),
      family_id=@catalog.AotKernelFamilyId::from_stable_int(1001),
    ),
    entry_point_id=candidate.entry_point_id(),
  )
  let bound = bind_promoted_paged_attention_parallel_release_candidate(
    candidate,
    entry,
    candidate.source_digest(),
    evidence,
  )
  assert_false(bound.manifest_bindable())
  assert_true(
    @utf8.decode(bound.binding_record_bytes()).contains(
      "promotion_evidence_sha256=" + evidence.digest().as_hex(),
    ),
  )
  try
    admit_paged_attention_parallel_promotion_evidence(
      candidate,
      parallel_promotion_trials(slow=true),
      parallel_end_to_end(),
      parallel_promotion_limits(),
    )
  catch {
    PagedAttentionParallelMismatch(MicrobenchmarkWin) => ()
    _ => fail("unexpected slow-candidate rejection")
  } noraise {
    _ => fail("slow candidate gained promotion evidence")
  }
}

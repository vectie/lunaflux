///|
fn paged_profile_test_plan() -> @plan.ModelPlan raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    context_length=64,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      spec,
      @spec.ContentDigest::from_sha256(
        "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::new(4),
  )
}

///|
fn paged_profile_page_table_trace(
  first : Char,
) -> @luna_profile_priority.LunaPagedPageTableTraceDigest raise {
  @luna_profile_priority.LunaPagedPageTableTraceDigest::from_sha256(
    String::make(64, first),
  )
}

///|
fn paged_profile_mixed_rows() -> Array[
  @luna_profile_priority.LunaPagedProfileRow,
] raise {
  [
    @luna_profile_priority.LunaPagedProfileRow::new(
      phase=PagedPrefill,
      query_length=4,
      context_length=4,
      cache_position=0,
      page_table_start=0,
      block_count=1,
    ),
    @luna_profile_priority.LunaPagedProfileRow::new(
      phase=PagedDecode,
      query_length=1,
      context_length=9,
      cache_position=8,
      page_table_start=1,
      block_count=3,
    ),
  ]
}

///|
fn paged_profile_shape(
  page_table_trace? : Char = 'e',
) -> @luna_profile_priority.LunaPagedProfileShape raise {
  @luna_profile_priority.LunaPagedProfileShape::new(
    tokens_per_page=4,
    prefill_row_count=1,
    decode_row_count=1,
    query_token_count=5,
    page_table_block_count=4,
    page_table_trace_digest=paged_profile_page_table_trace(page_table_trace),
    paged_profile_mixed_rows(),
  )
}

///|
fn paged_profile_observation(
  operation : Int,
  self_time : Int64,
  shape? : @luna_profile_priority.LunaPagedProfileShape,
) -> @luna_profile_priority.LunaPagedKernelProfileObservation raise {
  let exact_shape = match shape {
    Some(value) => value
    None => try! paged_profile_shape()
  }
  @luna_profile_priority.LunaPagedKernelProfileObservation::new(
    operation_id=@plan.OperationId::new(operation),
    shape=exact_shape,
    call_count=10L,
    self_time_ns=self_time,
    inclusive_time_ns=self_time + 100L,
  )
}

///|
fn admit_paged_profile_with(
  observations : ArrayView[
    @luna_profile_priority.LunaPagedKernelProfileObservation,
  ],
  declared_total : Int64,
  launches : @launch_contract.PagedKvLaunchContractSet,
  profile_id : Int,
  trace : Char,
) -> @luna_profile_priority.LunaPagedKernelProfileCapture raise {
  let plan = try! paged_profile_test_plan()
  let selected_profile = try! @launch_contract.KernelProfileId::from_stable_int(
    profile_id,
  )
  let workload = try! profile_test_digest('a')
  let profiler = try! profile_test_digest('b')
  let trace_digest = try! profile_test_digest(trace)
  let limits = try! @luna_profile_priority.LunaKernelProfileLimits::new(
    maximum_observations=16,
    maximum_canonical_bytes=16384,
  )
  @luna_profile_priority.admit_luna_paged_kernel_profile(
    plan,
    launches,
    profile_id=selected_profile,
    workload~,
    profiler~,
    trace=trace_digest,
    sampled_wall_time_ns=1000L,
    declared_attributed_self_time_ns=declared_total,
    observations,
    limits,
  )
}

///|
fn admit_paged_profile(
  observations : ArrayView[
    @luna_profile_priority.LunaPagedKernelProfileObservation,
  ],
  declared_total : Int64,
) -> @luna_profile_priority.LunaPagedKernelProfileCapture raise {
  let plan = try! paged_profile_test_plan()
  admit_paged_profile_with(
    observations,
    declared_total,
    try! paged_profile_launches(plan),
    7,
    'f',
  )
}

///|
test "paged profile binds exact mixed rows and selects deterministically" {
  let observations = [
    paged_profile_observation(3, 200L),
    paged_profile_observation(4, 700L),
  ]
  let first = admit_paged_profile(observations, 900L)
  let second = admit_paged_profile(observations, 900L)
  assert_eq(first, second)
  assert_eq(first.observation_count(), 2)
  assert_eq(first.priority().operation_id().as_int(), 4)
  assert_eq(first.priority().operation_kind(), CausalAttention)
  assert_eq(first.priority().attributed_self_time_ns(), 700L)
  assert_eq(first.priority().attributed_share_ppm(), 777777)
  assert_eq(first.contract().kv_head_count(), 2)
  assert_eq(first.contract().head_dimension(), 4)
  assert_eq(first.contract().kv_dtype(), BFloat16)
  assert_eq(first.contract().tokens_per_page(), 4)
  assert_eq(first.contract().total_page_count(), 32)
  assert_eq(first.contract().scope(), KvSubsequence)
  assert_eq(first.contract().catalog_version(), @catalog.CatalogVersion::v3())
  assert_eq(first.target(), profile_test_target())
  assert_eq(first.contract().kernel_profile().id().as_int(), 7)
  assert_eq(first.contract().kernel_profile().max_query_rows(), 4)
  assert_eq(first.contract().kernel_profile().max_query_tokens(), 16)
  assert_eq(first.contract().kernel_profile().max_page_table_entries(), 16)
  assert_eq(
    first.contract().device_kv_layout(),
    paged_profile_launches(paged_profile_test_plan()).device_kv_layout(),
  )
  assert_eq(first.trace_digest(), profile_test_digest('f'))
}

///|
test "paged profile tie retains immutable model-plan order" {
  let admitted = admit_paged_profile(
    [paged_profile_observation(3, 400L), paged_profile_observation(4, 400L)],
    800L,
  )
  assert_eq(admitted.priority().operation_id().as_int(), 3)
}

///|
test "paged observations cannot describe a stateless execution plan" {
  try
    @luna_profile_priority.admit_luna_paged_kernel_profile(
      profile_test_plan(),
      paged_profile_launches(paged_profile_test_plan()),
      profile_id=@launch_contract.KernelProfileId::from_stable_int(7),
      workload=profile_test_digest('a'),
      profiler=profile_test_digest('b'),
      trace=profile_test_digest('f'),
      sampled_wall_time_ns=1000L,
      declared_attributed_self_time_ns=600L,
      [paged_profile_observation(4, 600L)],
      @luna_profile_priority.LunaKernelProfileLimits::new(
        maximum_observations=16,
        maximum_canonical_bytes=16384,
      ),
    )
  catch {
    @luna_profile_priority.LunaKernelProfileUnsupported(ExecutionMode) => ()
    _ => fail("unexpected stateless-plan rejection")
  } noraise {
    _ => fail("paged profile evidence admitted a stateless plan")
  }
}

///|
test "paged profile shape copies caller rows and rejects aliases" {
  let rows = paged_profile_mixed_rows()
  let shape = @luna_profile_priority.LunaPagedProfileShape::new(
    tokens_per_page=4,
    prefill_row_count=1,
    decode_row_count=1,
    query_token_count=5,
    page_table_block_count=4,
    page_table_trace_digest=paged_profile_page_table_trace('e'),
    rows,
  )
  rows[0] = @luna_profile_priority.LunaPagedProfileRow::new(
    phase=PagedPrefill,
    query_length=3,
    context_length=3,
    cache_position=0,
    page_table_start=0,
    block_count=1,
  )
  assert_eq(shape, paged_profile_shape())
  let observations = [paged_profile_observation(4, 600L, shape~)]
  let capture = admit_paged_profile(observations, 600L)
  observations[0] = paged_profile_observation(3, 600L, shape~)
  assert_eq(capture.priority().operation_id().as_int(), 4)
  assert_eq(
    capture,
    admit_paged_profile([paged_profile_observation(4, 600L, shape~)], 600L),
  )
}

///|
test "paged profile shape rejects incomplete and inconsistent geometry" {
  try
    @luna_profile_priority.LunaPagedPageTableTraceDigest::from_sha256(
      "not-a-trace-digest",
    )
  catch {
    LunaKernelProfileInvalid(Digest) => ()
    _ => fail("invalid page-table trace digest returned the wrong rule")
  } noraise {
    _ => fail("invalid page-table trace digest was admitted")
  }
  try
    @luna_profile_priority.LunaPagedProfileRow::new(
      phase=PagedDecode,
      query_length=2,
      context_length=2,
      cache_position=0,
      page_table_start=0,
      block_count=1,
    )
  catch {
    LunaKernelProfileInvalid(Row) => ()
    _ => fail("invalid decode query returned the wrong rule")
  } noraise {
    _ => fail("multi-token decode row was admitted")
  }
  try
    @luna_profile_priority.LunaPagedProfileShape::new(
      tokens_per_page=4,
      prefill_row_count=2,
      decode_row_count=0,
      query_token_count=5,
      page_table_block_count=4,
      page_table_trace_digest=paged_profile_page_table_trace('e'),
      paged_profile_mixed_rows(),
    )
  catch {
    @luna_profile_priority.LunaKernelProfileMismatch(Shape) => ()
    _ => fail("incomplete row declaration returned the wrong rule")
  } noraise {
    _ => fail("incomplete row declaration was admitted")
  }
  let wrong_blocks = [
    @luna_profile_priority.LunaPagedProfileRow::new(
      phase=PagedPrefill,
      query_length=4,
      context_length=4,
      cache_position=0,
      page_table_start=0,
      block_count=2,
    ),
  ]
  try
    @luna_profile_priority.LunaPagedProfileShape::new(
      tokens_per_page=4,
      prefill_row_count=1,
      decode_row_count=0,
      query_token_count=4,
      page_table_block_count=2,
      page_table_trace_digest=paged_profile_page_table_trace('e'),
      wrong_blocks,
    )
  catch {
    @luna_profile_priority.LunaKernelProfileMismatch(PageTable) => ()
    _ => fail("wrong block geometry returned the wrong rule")
  } noraise {
    _ => fail("wrong block geometry was admitted")
  }
  try
    @luna_profile_priority.LunaPagedProfileShape::new(
      tokens_per_page=4,
      prefill_row_count=1,
      decode_row_count=1,
      query_token_count=5,
      page_table_block_count=4,
      page_table_trace_digest=paged_profile_page_table_trace('e'),
      [
        @luna_profile_priority.LunaPagedProfileRow::new(
          phase=PagedDecode,
          query_length=1,
          context_length=9,
          cache_position=8,
          page_table_start=0,
          block_count=3,
        ),
        @luna_profile_priority.LunaPagedProfileRow::new(
          phase=PagedPrefill,
          query_length=4,
          context_length=4,
          cache_position=0,
          page_table_start=3,
          block_count=1,
        ),
      ],
    )
  catch {
    @luna_profile_priority.LunaKernelProfileInvalid(RowOrder) => ()
    _ => fail("row phase order returned the wrong rule")
  } noraise {
    _ => fail("decode-before-prefill shape was admitted")
  }
}

///|
test "paged admission rejects foreign launch authority page and duplicate shape" {
  let valid = paged_profile_observation(4, 600L)
  let foreign_plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      @spec.LlamaModelSpec::new(
        vocabulary_size=128,
        hidden_size=16,
        intermediate_size=32,
        layer_count=2,
        attention_head_count=4,
        kv_head_count=2,
        context_length=64,
        rope_theta=10000.0,
        weight_dtype=BFloat16,
        kv_dtype=BFloat16,
      ),
      @spec.ContentDigest::from_sha256(
        "4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::new(4),
  )
  try
    @luna_profile_priority.admit_luna_paged_kernel_profile(
      foreign_plan,
      paged_profile_launches(paged_profile_test_plan()),
      profile_id=@launch_contract.KernelProfileId::from_stable_int(7),
      workload=profile_test_digest('a'),
      profiler=profile_test_digest('b'),
      trace=profile_test_digest('f'),
      sampled_wall_time_ns=1000L,
      declared_attributed_self_time_ns=600L,
      [valid],
      @luna_profile_priority.LunaKernelProfileLimits::new(
        maximum_observations=16,
        maximum_canonical_bytes=16384,
      ),
    )
  catch {
    @luna_profile_priority.LunaKernelProfileMismatch(Contract) => ()
    _ => fail("foreign launch authority returned the wrong rule")
  } noraise {
    _ => fail("foreign launch authority was admitted")
  }
  try
    admit_paged_profile_with(
      [valid],
      600L,
      paged_profile_launches(paged_profile_test_plan()),
      8,
      'f',
    )
  catch {
    @luna_profile_priority.LunaKernelProfileMismatch(Contract) => ()
    _ => fail("unknown launch profile returned the wrong rule")
  } noraise {
    _ => fail("unknown launch profile was admitted")
  }
  try admit_paged_profile([paged_profile_observation(1, 600L)], 600L) catch {
    @luna_profile_priority.LunaKernelProfileMismatch(Contract) => ()
    _ => fail("out-of-scope operation returned the wrong rule")
  } noraise {
    _ => fail("operation absent from the admitted launch scope was admitted")
  }
  let over_capacity = @luna_profile_priority.LunaPagedProfileShape::new(
    tokens_per_page=4,
    prefill_row_count=1,
    decode_row_count=0,
    query_token_count=17,
    page_table_block_count=5,
    page_table_trace_digest=paged_profile_page_table_trace('e'),
    [
      @luna_profile_priority.LunaPagedProfileRow::new(
        phase=PagedPrefill,
        query_length=17,
        context_length=17,
        cache_position=0,
        page_table_start=0,
        block_count=5,
      ),
    ],
  )
  try
    admit_paged_profile(
      [paged_profile_observation(4, 600L, shape=over_capacity)],
      600L,
    )
  catch {
    @luna_profile_priority.LunaKernelProfileMismatch(Shape) => ()
    _ => fail("profile-capacity substitution returned the wrong rule")
  } noraise {
    _ => fail("shape outside the admitted profile capacity was admitted")
  }
  let wrong_page_shape = @luna_profile_priority.LunaPagedProfileShape::new(
    tokens_per_page=8,
    prefill_row_count=1,
    decode_row_count=0,
    query_token_count=4,
    page_table_block_count=1,
    page_table_trace_digest=paged_profile_page_table_trace('e'),
    [
      @luna_profile_priority.LunaPagedProfileRow::new(
        phase=PagedPrefill,
        query_length=4,
        context_length=4,
        cache_position=0,
        page_table_start=0,
        block_count=1,
      ),
    ],
  )
  try
    admit_paged_profile(
      [paged_profile_observation(4, 600L, shape=wrong_page_shape)],
      600L,
    )
  catch {
    @luna_profile_priority.LunaKernelProfileMismatch(PageTable) => ()
    _ => fail("page-size substitution returned the wrong rule")
  } noraise {
    _ => fail("page-size substitution was admitted")
  }
  try admit_paged_profile([valid, valid], 1200L) catch {
    @luna_profile_priority.LunaKernelProfileInvalid(ObservationOrder) => ()
    _ => fail("duplicate shape returned the wrong rule")
  } noraise {
    _ => fail("duplicate paged shape was admitted")
  }
}

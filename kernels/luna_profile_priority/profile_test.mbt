///|
fn profile_test_plan() -> @plan.ModelPlan raise {
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
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  @llama.build(metadata)
}

///|
fn profile_test_digest(
  first : Char,
) -> @luna_profile_priority.LunaKernelProfileDigest raise {
  let value = String::make(64, first)
  @luna_profile_priority.LunaKernelProfileDigest::from_sha256(value)
}

///|
fn profile_test_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn profile_test_observation(
  operation : Int,
  tokens : Int,
  self_time : Int64,
) -> @luna_profile_priority.LunaKernelProfileObservation raise {
  @luna_profile_priority.LunaKernelProfileObservation::new(
    operation_id=@plan.OperationId::new(operation),
    shape=@device_profile.ExactExecutionShape::full_prefill(
      batch_rows=1,
      sequence_tokens=tokens,
    ),
    call_count=10L,
    self_time_ns=self_time,
    inclusive_time_ns=self_time + 100L,
  )
}

///|
test "profile priority is content-addressed and chooses measured self time" {
  let plan = profile_test_plan()
  let observations = [
    profile_test_observation(1, 8, 200L),
    profile_test_observation(4, 8, 600L),
    profile_test_observation(4, 16, 100L),
  ]
  let limits = @luna_profile_priority.LunaKernelProfileLimits::new(
    maximum_observations=16,
    maximum_canonical_bytes=4096,
  )
  let first = @luna_profile_priority.admit_luna_kernel_profile(
    plan,
    profile_test_target(),
    workload=profile_test_digest('a'),
    profiler=profile_test_digest('b'),
    sampled_wall_time_ns=1000L,
    declared_attributed_self_time_ns=900L,
    observations,
    limits,
  )
  let second = @luna_profile_priority.admit_luna_kernel_profile(
    plan,
    profile_test_target(),
    workload=profile_test_digest('a'),
    profiler=profile_test_digest('b'),
    sampled_wall_time_ns=1000L,
    declared_attributed_self_time_ns=900L,
    observations,
    limits,
  )
  assert_eq(first, second)
  assert_eq(first.observation_count(), 3)
  assert_eq(first.priority().operation_id().as_int(), 4)
  assert_eq(first.priority().operation_kind(), CausalAttention)
  assert_eq(first.priority().attributed_self_time_ns(), 700L)
  assert_eq(first.priority().attributed_share_ppm(), 777777)
}

///|
test "profile priority rejects order total shape and operation substitution" {
  let plan = profile_test_plan()
  let limits = @luna_profile_priority.LunaKernelProfileLimits::new(
    maximum_observations=16,
    maximum_canonical_bytes=4096,
  )
  let valid = profile_test_observation(4, 8, 600L)
  let cases = [
    ([valid, profile_test_observation(1, 8, 200L)], 800L),
    ([valid], 601L),
    ([profile_test_observation(4, 128, 600L)], 600L),
    ([profile_test_observation(99, 8, 600L)], 600L),
  ]
  for item in cases {
    let (observations, total) = item
    try
      @luna_profile_priority.admit_luna_kernel_profile(
        plan,
        profile_test_target(),
        workload=profile_test_digest('a'),
        profiler=profile_test_digest('b'),
        sampled_wall_time_ns=1000L,
        declared_attributed_self_time_ns=total,
        observations,
        limits,
      )
    catch {
      @luna_profile_priority.LunaKernelProfileInvalid(ObservationOrder)
      | @luna_profile_priority.LunaKernelProfileMismatch(Total)
      | @luna_profile_priority.LunaKernelProfileMismatch(Shape)
      | @luna_profile_priority.LunaKernelProfileMismatch(Operation) => ()
      _ => fail("unexpected profile rejection")
    } noraise {
      _ => fail("invalid profile capture was admitted")
    }
  }
}

///|
test "profile priority rejects targets without the Phase 5 BF16 capability" {
  let plan = profile_test_plan()
  let unsupported = @catalog.DeviceTarget::new(
    compute_major=7,
    compute_minor=5,
    supports_bf16=false,
    supports_cublas_lt=true,
  )
  try
    @luna_profile_priority.admit_luna_kernel_profile(
      plan,
      unsupported,
      workload=profile_test_digest('a'),
      profiler=profile_test_digest('b'),
      sampled_wall_time_ns=1000L,
      declared_attributed_self_time_ns=600L,
      [profile_test_observation(4, 8, 600L)],
      @luna_profile_priority.LunaKernelProfileLimits::new(
        maximum_observations=16,
        maximum_canonical_bytes=4096,
      ),
    )
  catch {
    @luna_profile_priority.LunaKernelProfileUnsupported(Target) => ()
    _ => fail("unexpected unsupported-target rejection")
  } noraise {
    _ => fail("unsupported profile target was admitted")
  }
}

///|
test "stateless profile evidence cannot describe a paged execution plan" {
  let paged = @llama.build_paged(
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
        "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
    ),
  )
  try
    @luna_profile_priority.admit_luna_kernel_profile(
      paged,
      profile_test_target(),
      workload=profile_test_digest('a'),
      profiler=profile_test_digest('b'),
      sampled_wall_time_ns=1000L,
      declared_attributed_self_time_ns=600L,
      [profile_test_observation(4, 8, 600L)],
      @luna_profile_priority.LunaKernelProfileLimits::new(
        maximum_observations=16,
        maximum_canonical_bytes=4096,
      ),
    )
  catch {
    @luna_profile_priority.LunaKernelProfileUnsupported(ExecutionMode) => ()
    _ => fail("unexpected paged-plan rejection")
  } noraise {
    _ => fail("stateless profile evidence admitted a paged plan")
  }
}

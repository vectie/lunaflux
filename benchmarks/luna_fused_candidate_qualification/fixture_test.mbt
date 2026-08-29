///|
fn fused_benchmark_test_set() -> @luna_fused_candidate_export.LunaFusedCandidateQualificationSet raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=256,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=4,
      max_arena_bytes=67108864L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=1,
    max_query_tokens=8,
    max_page_table_entries=4,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="b".repeat(64),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let reference = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
  @luna_fused_candidate_export.prepare_qualification_set(
    plan, layout, reference,
  )
}

///|
fn fused_benchmark_test_digest(
  digit : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    digit.repeat(64),
  )
}

///|
fn fused_benchmark_test_protocol(
  candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  shape_set : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  warmup_iterations : Int,
  trials_per_shape : Int,
  repetitions_per_arm : Int,
) -> FusedCandidateBenchmarkProtocol raise {
  FusedCandidateBenchmarkProtocol::new(
    candidate~,
    shape_set~,
    clock=fused_benchmark_test_digest("c"),
    resource_collector=fused_benchmark_test_digest("d"),
    warmup_iterations~,
    trials_per_shape~,
    repetitions_per_arm~,
  )
}

///|
fn fused_benchmark_test_shapes() -> Array[FusedCandidateBenchmarkShape] {
  [SingleOrigin, SinglePageTail, CrossPagePair, EightTokenPageBoundaries]
}

///|
fn fused_benchmark_test_shape_tokens(
  shape : FusedCandidateBenchmarkShape,
) -> Int {
  match shape {
    SingleOrigin | SinglePageTail => 1
    CrossPagePair => 2
    EightTokenPageBoundaries => 8
  }
}

///|
fn fused_benchmark_test_observations(
  dispatch_canary : UInt,
  baseline_duration_ns? : Int64 = 1000L,
  candidate_duration_ns? : Int64 = 900L,
  corrupt_canary? : Bool = false,
  unbalanced_resources? : Bool = false,
  wrong_pair_order? : Bool = false,
) -> Array[FusedCandidateBenchmarkPairedObservation] raise {
  let output : Array[FusedCandidateBenchmarkPairedObservation] = []
  let repetitions = fused_candidate_benchmark_repetitions_per_arm()
  for shape in fused_benchmark_test_shapes() {
    let tokens = fused_benchmark_test_shape_tokens(shape)
    let work = Int64::from_int(repetitions * tokens)
    let expected_delta = dispatch_canary *
      repetitions.reinterpret_as_uint() *
      tokens.reinterpret_as_uint()
    for trial in 0..<fused_candidate_benchmark_trials_per_shape() {
      let canary_corruption : UInt = if corrupt_canary && trial == 0 {
        1
      } else {
        0
      }
      let baseline = FusedCandidateBenchmarkArmObservation::new(
        duration_ns=baseline_duration_ns + Int64::from_int(trial),
        work_units=work,
        resources_acquired=2L,
        resources_released=if unbalanced_resources && trial == 0 {
          1L
        } else {
          2L
        },
        live_resources_before=4L,
        live_resources_after=4L,
        live_bytes_before=4096L,
        live_bytes_after=4096L,
        resource_failures=0L,
        dispatch_canary_before=11,
        dispatch_canary_after=11,
      )
      let candidate = FusedCandidateBenchmarkArmObservation::new(
        duration_ns=candidate_duration_ns + Int64::from_int(trial),
        work_units=work,
        resources_acquired=2L,
        resources_released=2L,
        live_resources_before=4L,
        live_resources_after=4L,
        live_bytes_before=4096L,
        live_bytes_after=4096L,
        resource_failures=0L,
        dispatch_canary_before=17,
        dispatch_canary_after=17 + expected_delta + canary_corruption,
      )
      output.push(
        FusedCandidateBenchmarkPairedObservation::new(
          shape~,
          trial_ordinal=trial,
          repetitions~,
          order=if wrong_pair_order && trial == 0 {
            CandidateThenBaseline
          } else if trial % 2 == 0 {
            BaselineThenCandidate
          } else {
            CandidateThenBaseline
          },
          baseline~,
          candidate~,
        ),
      )
    }
  }
  output
}

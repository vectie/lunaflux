///|
fn ingress_policy_fixture() -> ReadOnlyArray[
  @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
] raise {
  let model = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=1024,
    hidden_size=512,
    intermediate_size=1024,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=128,
    context_length=256,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  let plan = @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("3".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(4),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=64,
      max_arena_bytes=67108864L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(19),
    max_query_rows=4,
    max_query_tokens=8,
    max_page_table_entries=16,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
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
  let set = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
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
  let candidates = set.candidates()
  guard candidates[2] is Projection(qkv) &&
    candidates[3] is Pointwise(norm) &&
    candidates[4] is Pointwise(rope) &&
    candidates[5] is PagedAttention(attention) else {
    fail("missing fixture operations")
  }
  let operands = attention
    .operands()
    .filter(operand => {
      match operand.role() {
        ActivationOutput(_) => false
        _ => true
      }
    })
  let kv_write = @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
    plan.operations()[5],
    profile,
    layout,
    @catalog.AotKernelEntryPointId::from_stable_int(3901),
    target,
    compiler,
    operands[:],
    @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
  )
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
    maximum_source_bytes=1048576,
    maximum_recipe_bytes=65536,
    maximum_promotion_bytes=65536,
  )
  let baseline = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_candidate(
    plan, layout, qkv, norm, rope, kv_write, target, compiler, limits,
  )
  let production = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
    baseline, limits,
  )
  prepare_qwen_ingress_resource_variants(
    plan, layout, qkv, norm, rope, kv_write, set, baseline, production,
  )
}

///|
test "ingress exporter uses measured resource policy and retains source-bound unmeasured default" {
  let variants = ingress_policy_fixture()
  // Finite backend domain: rows × stages × transfer × lookahead × columns.
  assert_true(variants.length() > 2 && variants.length() <= 4 * 3 * 5 * 2 * 2)
  let baseline = variants[0]
  let alternate = variants
    .iter()
    .find_first(v => {
      let p = v.ingress_policy()
      p.live_accumulators() == 1 &&
      p.operand_stages() == 3 &&
      p.transfer_tiles() == 2 &&
      p.fragment_stages() == 2
    })
    .unwrap()
  assert_eq(baseline.ingress_policy().live_accumulators(), 2)
  assert_eq(alternate.ingress_policy().live_accumulators(), 1)
  assert_true(
    @utf8.decode(baseline.source_bytes()).contains("#define LF_LIVE_COLUMNS 2"),
  )
  assert_true(
    @utf8.decode(alternate.source_bytes()).contains("#define LF_LIVE_COLUMNS 1"),
  )
  assert_true(
    @utf8.decode(alternate.source_bytes()).contains("ingress_operands[13824]"),
  )
  let scope = baseline
    .resource_scope(device="fixture", toolchain="tool-v1")
    .unwrap()
  let (device, toolchain, shape, program) = scope.identity()
  let prefix = "luna-projection-resources-v1\nscope\t\{device}\t\{toolchain}\t\{shape}\t\{program}\nbudget\t128\t49152\t128\t1\n"
  let base = "record\t\{baseline.ingress_policy().id()}\t\{baseline.source_digest().as_hex()}\t100\t5\t64\t1\n"
  let fast = "record\t\{alternate.ingress_policy().id()}\t\{alternate.source_digest().as_hex()}\t50\t5\t48\t2\n"
  let (selected, origin) = select_qwen_ingress_resource_variant(
    baseline,
    variants[:],
    Some((@utf8.encode(prefix + base + fast), device, toolchain)),
  )
  assert_eq(selected.source_digest(), alternate.source_digest())
  assert_eq(origin, Measured(latency_ns=50L, samples=5))
  let exact_prefix = prefix.replace(old="resources-v1", new="resources-v2") +
    "workload\tprefill\t8\t1\t0\t8\t4\nprovenance\tGPU-fixture\n"
  let (workload_selected, workload_origin) = select_qwen_ingress_resource_variant(
    baseline,
    variants[:],
    Some((@utf8.encode(exact_prefix + base + fast), device, toolchain)),
  )
  assert_eq(workload_selected.source_digest(), alternate.source_digest())
  assert_eq(workload_origin, Measured(latency_ns=50L, samples=5))
  let stale = "record\t\{baseline.ingress_policy().id()}\t\{"a".repeat(64)}\t100\t5\t64\t1\n"
  let (default, default_origin) = select_qwen_ingress_resource_variant(
    baseline,
    variants[:],
    None,
  )
  assert_eq(default.source_digest(), baseline.source_digest())
  assert_eq(default_origin, UnmeasuredPreferred)
  for
    input in [
      Some((@utf8.encode(prefix + fast), device, toolchain)),
      Some((@utf8.encode(prefix + stale + fast), device, toolchain)),
    ] {
    try
      select_qwen_ingress_resource_variant(baseline, variants[:], input)
    catch {
      QwenIngressPreparation => ()
      _ => fail("unexpected policy selection error")
    } noraise {
      _ => fail("missing measured baseline bypassed resource envelope")
    }
  }
  let excessive = "record\t\{alternate.ingress_policy().id()}\t\{alternate.source_digest().as_hex()}\t1\t5\t129\t2\n"
  let (selected, origin) = select_qwen_ingress_resource_variant(
    baseline,
    variants[:],
    Some((@utf8.encode(prefix + base + excessive), device, toolchain)),
  )
  assert_eq(selected.source_digest(), baseline.source_digest())
  assert_eq(origin, Measured(latency_ns=100L, samples=5))
}

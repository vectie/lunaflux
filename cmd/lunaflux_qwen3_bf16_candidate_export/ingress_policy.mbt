///|
fn prepare_qwen_ingress_resource_variants(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  baseline : @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
  production : @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
  maximum_shared_bytes? : Int64 = 49152L,
) -> ReadOnlyArray[
  @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
] raise Qwen3Bf16CandidateExportCliError {
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
    maximum_source_bytes=1048576,
    maximum_recipe_bytes=65536,
    maximum_promotion_bytes=65536,
  ) catch {
    _ => raise QwenIngressPreparation
  }
  let variants = [production]
  for
    policy in @luna_cuda_fused_parallel_aot.qwen_ingress_policies(
      baseline.tile_compilation(),
      maximum_shared_bytes~,
    ) {
    if policy == production.ingress_policy() {
      continue
    }
    let qualification = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_candidate(
      plan,
      layout,
      qkv,
      qk_norm,
      rope,
      kv_write,
      set.target(),
      set.compiler_policy(),
      limits,
      ingress_policy=policy,
      maximum_shared_bytes~,
    ) catch {
      _ => raise QwenIngressPreparation
    }
    variants.push(
      @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
        qualification, limits,
      ) catch {
        _ => raise QwenIngressPreparation
      },
    )
  }
  ReadOnlyArray::from_array(variants)
}

///|
fn select_qwen_ingress_resource_variant(
  baseline : @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
  variants : ArrayView[
    @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
  ],
  input : (Bytes, String, String)?,
) -> (
  @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
  @resource_policy.SelectionOrigin,
) raise Qwen3Bf16CandidateExportCliError {
  guard input is Some((bytes, device, toolchain)) else {
    return (baseline, UnmeasuredPreferred)
  }
  guard baseline.resource_scope(device~, toolchain~) is Some(scope) else {
    raise QwenIngressPreparation
  }
  let observations = @luna_projection_tuning.parse_projection_resource_records(
    bytes,
    scope~,
  ) catch {
    _ => raise QwenIngressPreparation
  }
  let scope = @luna_projection_tuning.projection_resource_selection_scope(
    bytes,
    scope~,
    max_query_tokens=baseline.profile().max_query_tokens(),
    max_batch_rows=baseline.profile().max_query_rows(),
  ) catch {
    _ => raise QwenIngressPreparation
  }
  let observed_budget = @luna_projection_tuning.parse_projection_resource_budget(
    bytes,
  ) catch {
    _ => raise QwenIngressPreparation
  }
  let budget = @luna_cuda_fused_parallel_aot.qwen_ingress_resource_budget(
    maximum_shared_bytes=Int64::min(98304L, observed_budget.max_shared_bytes()),
  ).intersect(observed_budget)
  // Do not replace an unmeasured baseline solely because some alternative has
  // a measurement. A matched baseline is needed for a comparable selection.
  guard @resource_policy.select(
      scope,
      budget,
      [baseline.resource_candidate()],
      observations[:],
      preferred_id=baseline.ingress_policy().id(),
    )
    is Some(reference) else {
    raise QwenIngressPreparation
  }
  guard reference.origin() is Measured(..) else {
    return (baseline, UnmeasuredPreferred)
  }
  // A larger target frontier is not a larger claim of measured coverage.
  // Select only identities present in the bounded, exact-artifact table.
  let measured_identities = observations.map(o => o.identity())
  let candidates = variants
    .filter(v => {
      measured_identities.contains(
        (v.ingress_policy().id(), v.source_digest().as_hex()),
      )
    })
    .map(v => v.resource_candidate())
  guard @resource_policy.select(
      scope,
      budget,
      candidates,
      observations[:],
      preferred_id=baseline.ingress_policy().id(),
    )
    is Some(selection) else {
    raise QwenIngressPreparation
  }
  for variant in variants {
    if variant.ingress_policy() == selection.policy() &&
      variant.source_digest().as_hex() == selection.artifact() {
      return (variant, selection.origin())
    }
  }
  raise QwenIngressPreparation
}

///|
fn v3_policy_tag(
  policy : @numeric_contract.ActivationScalePolicy,
) -> String raise Fp8ReusablePagedRecipeV3Error {
  if policy ==
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() {
    "dynamic-per-tensor-f32-v1"
  } else if policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() {
    "gated-external-post-silu-bound-expf-f32-v2"
  } else {
    raise InvalidFp8ReusablePagedRecipeV3(LaunchAuthority)
  }
}

///|
fn v3_operation_tag(kind : @plan.OperationKind) -> String {
  match kind {
    QkvProjection => "qkv-projection"
    OutputProjection => "output-projection"
    GatedMlp => "gated-mlp"
    LanguageModelHead => "language-model-head"
    Embedding => "embedding"
    RmsNorm => "rms-norm"
    QkRmsNorm => "qk-rms-norm"
    RotaryEmbedding => "rotary-embedding"
    CausalAttention => "causal-attention"
    ResidualAdd => "residual-add"
  }
}

///|
fn encode_reusable_paged_v3(
  source : Fp8RuntimeRecipeV2,
  profile : @launch_contract.PagedKernelProfile,
  kv_layout : @device_layout.DeviceKvLayout,
  launches : ReadOnlyArray[Fp8ReusablePagedLaunchV3],
  workspace_capacity_bytes : Int64,
  maximum : Int,
) -> Bytes raise Fp8ReusablePagedRecipeV3Error {
  let output : Array[Byte] = []
  let append = value => {
    append_v2_text(output, value, maximum) catch {
      _ => raise InvalidFp8ReusablePagedRecipeV3(CanonicalBytes)
    }
  }
  let append_region = region => {
    append_v2_region(output, region, maximum) catch {
      _ => raise InvalidFp8ReusablePagedRecipeV3(CanonicalBytes)
    }
  }
  append("lunaflux.fp8-reusable-paged-recipe.v3")
  let plan = source.startup().model_plan()
  append(plan.identity().content().as_hex())
  append(plan.identity().plan().as_hex())
  append(plan.numeric_binding().digest().as_hex())
  let target = source.startup().kernel_artifacts().manifest().target()
  append(target.compute_major().to_string())
  append(target.compute_minor().to_string())
  append(profile.id().as_int().to_string())
  append(profile.max_query_rows().to_string())
  append(profile.max_query_tokens().to_string())
  append(profile.max_page_table_entries().to_string())
  append(plan.shape_constraints().max_batch_rows().to_string())
  append(plan.shape_constraints().max_sequence_tokens().to_string())
  append(kv_layout.stable_layout_version().to_string())
  append(kv_layout.tokens_per_page().to_string())
  append(kv_layout.total_page_count().to_string())
  append(kv_layout.arena_bytes().to_string())
  append(source.weight_inspection().artifact_digest().as_hex())
  append(workspace_capacity_bytes.to_string())
  append(launches.length().to_string())
  for index, reusable in launches {
    let launch = reusable.source
    append(index.to_string())
    append(launch.operation().as_int().to_string())
    append(v3_operation_tag(launch.operation_kind()))
    append(launch.execution_digest().as_hex())
    append(v3_policy_tag(launch.activation_scale_policy()))
    append(launch.manifest_digest().as_hex())
    append(launch.module_digest().sha256())
    append(launch.entry_point().family().family_id().as_int().to_string())
    append(launch.entry_point().entry_point_id().as_int().to_string())
    append_v2_raw_launch_authority(output, launch, maximum) catch {
      _ => raise InvalidFp8ReusablePagedRecipeV3(CanonicalBytes)
    }
    let binding = source.weight_bindings()[index]
    append(binding.operation().as_int().to_string())
    for region in binding.regions() {
      append(region.tensor().as_int().to_string())
      append(region.storage_digest().as_hex())
      append_region(region.region())
    }
    for scale in launch.scale_tensor_regions() {
      append(scale.tensor().tensor().as_int().to_string())
      append(scale.storage_digest().as_hex())
      append_region(scale.region())
    }
    append(reusable.workspace_offset().to_string())
    append(reusable.workspace_bytes().to_string())
    append(reusable.workspace_alignment().to_string())
    append(reusable.stage_count().to_string())
    for ordinal in 0..<reusable.stage_count() {
      guard reusable.stage_at(ordinal) is Some(stage) else {
        raise InvalidFp8ReusablePagedRecipeV3(LaunchAuthority)
      }
      append(ordinal.to_string())
      let tag = if stage ==
        @numeric_contract.ActivationScaleStage::external_operation_input_v1() {
        "external-operation-input-v1"
      } else if stage ==
        @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1() {
        "post-silu-gate-up-product-v1"
      } else {
        raise InvalidFp8ReusablePagedRecipeV3(LaunchAuthority)
      }
      append(tag)
      append(
        (reusable.workspace_offset() + ordinal.to_int64() * 4L).to_string(),
      )
    }
  }
  Bytes::from_array(output)
}

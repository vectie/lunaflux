///|
fn release_target() -> @catalog.DeviceTarget raise Qwen3Bf16ReleaseBindError {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn release_layout(
  model_plan : @plan.ModelPlan,
  arguments : Qwen3ReleaseBindArguments,
) -> @device_layout.DeviceKvLayout raise Qwen3Bf16ReleaseBindError {
  @device_layout.plan(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=arguments.tokens_per_page,
      total_page_count=arguments.total_page_count,
      max_arena_bytes=17179869184L,
    ),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn release_profile(
  arguments : Qwen3ReleaseBindArguments,
) -> @launch_contract.PagedKernelProfile raise Qwen3Bf16ReleaseBindError {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=arguments.max_query_rows,
    max_query_tokens=arguments.max_query_tokens,
    max_page_table_entries=arguments.max_page_table_entries,
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn release_compiler(
  arguments : Qwen3ReleaseBindArguments,
) -> @luna_specialization.LunaCompilerPolicy raise Qwen3Bf16ReleaseBindError {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=arguments.toolchain_sha256,
      major=arguments.compiler_major.reinterpret_as_int(),
      minor=arguments.compiler_minor.reinterpret_as_int(),
      patch=arguments.compiler_patch.reinterpret_as_int(),
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn release_candidates(
  inputs : Qwen3ReleaseBindInputs,
  arguments : Qwen3ReleaseBindArguments,
) -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise Qwen3Bf16ReleaseBindError {
  let profile = release_profile(arguments)
  if profile.max_query_rows() !=
    inputs.plan.shape_constraints().max_batch_rows() ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_query_tokens().to_int64() >
    profile.max_query_rows().to_int64() *
    inputs.plan.shape_constraints().max_sequence_tokens().to_int64() {
    raise KernelBinding
  }
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    inputs.plan,
    release_layout(inputs.plan, arguments),
    profile,
    release_target(),
    release_compiler(arguments),
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=4096,
      max_source_bytes=16777216,
      max_recipe_bytes=16777216,
      max_total_bytes=1073741824,
    ),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn bundle_limits() -> @luna_kernel_bundle.LunaKernelBundleLimits raise Qwen3Bf16ReleaseBindError {
  @luna_kernel_bundle.LunaKernelBundleLimits::new(
    max_module_bytes=4194304L,
    max_total_module_bytes=2147483647L,
    max_modules=4096,
    max_operations=4096,
    manifest=@full_graph_manifest.FullGraphManifestLimits::new(
      max_manifest_bytes=1048576L,
      artifact_file=@artifact_file.KernelArtifactFileLimits::new(
        max_manifest_bytes=1048576L,
        max_json_depth=24,
        max_path_bytes=128,
        artifact=@artifact.KernelArtifactLimits::new(
          max_modules=4096,
          max_entry_points=4096,
          max_module_bytes=4194304L,
          max_total_module_bytes=2147483647L,
          max_function_symbol_bytes=96,
        ),
      ),
    ),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn producer_limits() -> @luna_bf16_kernel_producer.LunaBf16KernelProducerLimits raise Qwen3Bf16ReleaseBindError {
  @luna_bf16_kernel_producer.LunaBf16KernelProducerLimits::new(
    max_operations=4096,
    launch_contracts=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=4096,
      max_operands_per_contract=32,
    ),
    bundle=bundle_limits(),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

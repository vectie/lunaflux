///|
fn release_target() -> @catalog.DeviceTarget raise LunaBf16ReleaseBindError {
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
) -> @device_layout.DeviceKvLayout raise LunaBf16ReleaseBindError {
  @device_layout.plan(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=RELEASE_TOKENS_PER_PAGE,
      total_page_count=RELEASE_TOTAL_PAGE_COUNT,
      max_arena_bytes=1048576L,
    ),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn release_profile() -> @launch_contract.PagedKernelProfile raise LunaBf16ReleaseBindError {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=1,
    max_page_table_entries=RELEASE_MAX_PAGE_TABLE_ENTRIES,
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn release_compiler(
  arguments : ReleaseBindArguments,
) -> @luna_specialization.LunaCompilerPolicy raise LunaBf16ReleaseBindError {
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
  inputs : ReleaseBindInputs,
  arguments : ReleaseBindArguments,
) -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise LunaBf16ReleaseBindError {
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    inputs.plan,
    release_layout(inputs.plan),
    release_profile(),
    release_target(),
    release_compiler(arguments),
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn bundle_limits() -> @luna_kernel_bundle.LunaKernelBundleLimits raise LunaBf16ReleaseBindError {
  @luna_kernel_bundle.LunaKernelBundleLimits::new(
    max_module_bytes=4194304L,
    max_total_module_bytes=67108864L,
    max_modules=32,
    max_operations=64,
    manifest=@full_graph_manifest.FullGraphManifestLimits::new(
      max_manifest_bytes=1048576L,
      artifact_file=@artifact_file.KernelArtifactFileLimits::new(
        max_manifest_bytes=1048576L,
        max_json_depth=24,
        max_path_bytes=128,
        artifact=@artifact.KernelArtifactLimits::new(
          max_modules=32,
          max_entry_points=64,
          max_module_bytes=4194304L,
          max_total_module_bytes=67108864L,
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
fn producer_limits() -> @luna_bf16_kernel_producer.LunaBf16KernelProducerLimits raise LunaBf16ReleaseBindError {
  @luna_bf16_kernel_producer.LunaBf16KernelProducerLimits::new(
    max_operations=64,
    launch_contracts=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=64,
      max_operands_per_contract=16,
    ),
    bundle=bundle_limits(),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

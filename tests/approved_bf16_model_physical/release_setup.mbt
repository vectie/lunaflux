///|
const APPROVED_CONTEXT_TOKENS : Int = 256

///|
const APPROVED_TOKENS_PER_PAGE : Int = 8

///|
const APPROVED_CONTEXT_PAGE_COUNT : Int = 32

///|
fn approved_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn approved_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=1,
    max_page_table_entries=APPROVED_CONTEXT_PAGE_COUNT,
  )
}

///|
fn approved_kv_layout(
  plan : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=APPROVED_TOKENS_PER_PAGE,
      total_page_count=APPROVED_CONTEXT_PAGE_COUNT,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn approved_compiler_policy(
  toolchain_sha256 : String,
  compiler_major : UInt,
  compiler_minor : UInt,
  compiler_patch : UInt,
) -> @luna_specialization.LunaCompilerPolicy raise {
  if compiler_major == 0U ||
    compiler_major > 999999U ||
    compiler_minor > 999999U ||
    compiler_patch > 999999U {
    raise InvalidInput("compiler version component is out of bounds")
  }
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=toolchain_sha256,
      major=compiler_major.reinterpret_as_int(),
      minor=compiler_minor.reinterpret_as_int(),
      patch=compiler_patch.reinterpret_as_int(),
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn approved_candidate_set(
  inputs : ApprovedBf16Inputs,
  toolchain_sha256 : String,
  compiler_major : UInt,
  compiler_minor : UInt,
  compiler_patch : UInt,
) -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise {
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    inputs.plan,
    approved_kv_layout(inputs.plan),
    approved_profile(),
    approved_target(),
    approved_compiler_policy(
      toolchain_sha256, compiler_major, compiler_minor, compiler_patch,
    ),
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
}

///|
fn approved_kernel_bundle_limits() -> @luna_kernel_bundle.LunaKernelBundleLimits raise {
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
  )
}

///|
fn approved_producer_limits() -> @luna_bf16_kernel_producer.LunaBf16KernelProducerLimits raise {
  @luna_bf16_kernel_producer.LunaBf16KernelProducerLimits::new(
    max_operations=64,
    launch_contracts=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=64,
      max_operands_per_contract=16,
    ),
    bundle=approved_kernel_bundle_limits(),
  )
}

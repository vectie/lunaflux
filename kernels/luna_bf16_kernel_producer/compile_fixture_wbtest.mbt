///|
fn producer_fixture_candidate(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  legacy? : Bool = false,
) -> LunaBf16KernelCandidate raise {
  let profile = producer_fixture_profile()
  let layout = producer_fixture_layout()
  let operands = producer_fixture_operands(plan, operation, profile, layout)
  match operation.kind() {
    Embedding | RmsNorm | QkRmsNorm | RotaryEmbedding | ResidualAdd => {
      let candidate = if legacy {
        @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
          operation,
          profile,
          @catalog.AotKernelEntryPointId::from_stable_int(1),
          producer_fixture_target(),
          producer_fixture_compiler(),
          operands,
          @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
            max_source_bytes=65536,
            max_recipe_bytes=65536,
          ),
        )
      } else {
        @luna_cuda_pointwise_aot.lower_reusable_pointwise_cuda_aot_candidate(
          operation,
          profile,
          producer_fixture_target(),
          producer_fixture_compiler(),
          operands,
          @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
            max_source_bytes=65536,
            max_recipe_bytes=65536,
          ),
        )
      }
      Pointwise(candidate)
    }
    QkvProjection | OutputProjection | GatedMlp | LanguageModelHead => {
      let candidate = if legacy {
        @luna_cuda_projection_aot.lower_projection_cuda_aot_candidate(
          operation,
          profile,
          @catalog.AotKernelEntryPointId::from_stable_int(1),
          producer_fixture_target(),
          producer_fixture_compiler(),
          operands,
          @luna_cuda_projection_aot.ProjectionAotLimits::new(
            max_source_bytes=65536,
            max_recipe_bytes=65536,
          ),
        )
      } else {
        @luna_cuda_projection_aot.lower_reusable_projection_cuda_aot_candidate(
          operation,
          profile,
          producer_fixture_target(),
          producer_fixture_compiler(),
          operands,
          @luna_cuda_projection_aot.ProjectionAotLimits::new(
            max_source_bytes=65536,
            max_recipe_bytes=65536,
          ),
        )
      }
      Projection(candidate)
    }
    CausalAttention =>
      PagedAttention(
        @luna_cuda_paged_attention_aot.lower_reusable_paged_attention_cuda_aot_candidate(
          operation,
          profile,
          layout,
          producer_fixture_target(),
          producer_fixture_compiler(),
          operands,
          @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
            max_source_bytes=65536,
            max_recipe_bytes=65536,
          ),
        ),
      )
  }
}

///|
fn fixture_candidate_source(candidate : LunaBf16KernelCandidate) -> Bytes {
  match candidate {
    Pointwise(value) => value.source_bytes()
    Projection(value) => value.source_bytes()
    PagedAttention(value) => value.source_bytes()
  }
}

///|
fn fixture_candidate_recipe_digest(
  candidate : LunaBf16KernelCandidate,
) -> String {
  match candidate {
    Pointwise(value) => value.recipe_digest().as_hex()
    Projection(value) => value.recipe_digest().as_hex()
    PagedAttention(value) => value.recipe_digest().as_hex()
  }
}

///|
fn producer_fixture_compilation(
  candidate : LunaBf16KernelCandidate,
) -> LunaBf16OfflineCompilation raise {
  let module_bytes = fixture_candidate_source(candidate)
  let digest = sha256_bytes(module_bytes)
  LunaBf16OfflineCompilation::new(
    candidate~,
    first_module=module_bytes,
    second_module=module_bytes,
    receipt=@luna_kernel_bundle.LunaDeterministicCompileReceipt::from_offline_builder(
      source_digest_sha256=candidate_source_digest(candidate),
      recipe_digest_sha256=fixture_candidate_recipe_digest(candidate),
      toolchain_digest_sha256="3".repeat(64),
      driver_identity_sha256="4".repeat(64),
      first_artifact_digest_sha256=digest,
      second_artifact_digest_sha256=digest,
    ),
  )
}

///|
fn producer_fixture_compilations(
  legacy_operation? : Int = -1,
) -> Array[LunaBf16OfflineCompilation] raise {
  let plan = producer_fixture_plan()
  let result : Array[LunaBf16OfflineCompilation] = []
  for operation in plan.operations() {
    result.push(
      producer_fixture_compilation(
        producer_fixture_candidate(
          plan,
          operation,
          legacy=operation.id().as_int() == legacy_operation,
        ),
      ),
    )
  }
  result
}

///|
fn producer_fixture_limits() -> LunaBf16KernelProducerLimits raise {
  LunaBf16KernelProducerLimits::new(
    max_operations=64,
    launch_contracts=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=64,
      max_operands_per_contract=16,
    ),
    bundle=@luna_kernel_bundle.LunaKernelBundleLimits::new(
      max_module_bytes=65536L,
      max_total_module_bytes=589824L,
      max_modules=16,
      max_operations=64,
      manifest=@full_graph_manifest.FullGraphManifestLimits::new(
        max_manifest_bytes=1048576L,
        artifact_file=@artifact_file.KernelArtifactFileLimits::new(
          max_manifest_bytes=1048576L,
          max_json_depth=16,
          max_path_bytes=96,
          artifact=@artifact.KernelArtifactLimits::new(
            max_modules=16,
            max_entry_points=32,
            max_module_bytes=65536L,
            max_total_module_bytes=589824L,
            max_function_symbol_bytes=96,
          ),
        ),
      ),
    ),
  )
}

///|
fn fused_selection_test_digest(
  digit : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    digit.repeat(64),
  )
}

///|
fn fused_selection_test_target(minor? : Int = 0) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn fused_selection_test_launch(
  block_x? : Int = 128,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=8,
    grid_y=1,
    grid_z=1,
    block_x~,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn fused_selection_test_entry() -> ApprovedFusedKernelModule raise {
  let fallbacks : Array[FusedStandaloneFallbackIdentity] = []
  {
    family: QkvPositionedRopePagedKvWrite,
    candidate: fused_selection_test_digest("1"),
    source: fused_selection_test_digest("2"),
    recipe: fused_selection_test_digest("3"),
    toolchain: "4".repeat(64),
    driver: "5".repeat(64),
    target: fused_selection_test_target(),
    launch: fused_selection_test_launch(),
    symbol: "lunaflux_fused_qkv_rope_paged_kv_write_bf16_v1",
    abi: QkvPositionedRopePagedKvWriteAbiV1,
    purpose: QualificationDiagnostics,
    dispatch_canary_per_token: 17U,
    artifact_bytes: b"fixture-only-cubin",
    module_digest: fused_selection_test_digest("6"),
    receipt_digest: fused_selection_test_digest("7"),
    binding_digest: fused_selection_test_digest("8"),
    benchmark_digest: fused_selection_test_digest("9"),
    fallbacks: ReadOnlyArray::from_array(fallbacks),
  }
}

///|
fn fused_selection_test_exact(
  entry : ApprovedFusedKernelModule,
  family? : @luna_cuda_fused_parallel_aot.FusedParallelFamily = QkvPositionedRopePagedKvWrite,
  candidate? : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  source? : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe? : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  toolchain? : String = "4".repeat(64),
  driver? : String = "5".repeat(64),
  symbol? : String = "lunaflux_fused_qkv_rope_paged_kv_write_bf16_v1",
  abi? : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi = QkvPositionedRopePagedKvWriteAbiV1,
  launch? : @launch_contract.AotLaunchDimensions,
  module_digest? : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  binding? : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> Unit raise {
  fused_require_exact_selection(
    entry,
    family,
    candidate.unwrap_or(entry.candidate),
    source.unwrap_or(entry.source),
    recipe.unwrap_or(entry.recipe),
    toolchain,
    driver,
    symbol,
    abi,
    launch.unwrap_or(entry.launch),
    module_digest.unwrap_or(entry.module_digest),
    binding.unwrap_or(entry.binding_digest),
  )
}

///|
test "exact fused selection accepts only the complete bound identity" {
  fused_selection_test_exact(fused_selection_test_entry())
}

///|
test "hostile fused selection rejects candidate source recipe and family substitution" {
  let entry = fused_selection_test_entry()
  for selector in 0..<4 {
    try {
      match selector {
        0 =>
          fused_selection_test_exact(
            entry,
            candidate=fused_selection_test_digest("a"),
          )
        1 =>
          fused_selection_test_exact(
            entry,
            source=fused_selection_test_digest("a"),
          )
        2 =>
          fused_selection_test_exact(
            entry,
            recipe=fused_selection_test_digest("a"),
          )
        _ => fused_selection_test_exact(entry, family=ResidualRmsNorm)
      }
    } catch {
      MismatchedFusedArtifact(Candidate | Family) => ()
      _ => fail("unexpected fused identity substitution rejection")
    } noraise {
      _ => fail("fused identity substitution reached selection")
    }
  }
}

///|
test "hostile fused selection rejects compiler launch symbol ABI module and binding substitution" {
  let entry = fused_selection_test_entry()
  for selector in 0..<7 {
    try {
      match selector {
        0 => fused_selection_test_exact(entry, toolchain="a".repeat(64))
        1 => fused_selection_test_exact(entry, driver="a".repeat(64))
        2 => fused_selection_test_exact(entry, symbol="substituted")
        3 => fused_selection_test_exact(entry, abi=ResidualRmsNormAbiV1)
        4 =>
          fused_selection_test_exact(
            entry,
            launch=fused_selection_test_launch(block_x=64),
          )
        5 =>
          fused_selection_test_exact(
            entry,
            module_digest=fused_selection_test_digest("a"),
          )
        _ =>
          fused_selection_test_exact(
            entry,
            binding=fused_selection_test_digest("a"),
          )
      }
    } catch {
      MismatchedFusedArtifact(
        Compiler
        | Symbol
        | KernelAbi
        | Launch
        | Module
        | Binding
      ) => ()
      _ => fail("unexpected fused compiled identity rejection")
    } noraise {
      _ => fail("compiled identity substitution reached selection")
    }
  }
}

///|
test "hostile fused catalog authority rejects manifest replay and cross target" {
  let admitted = fused_selection_test_digest("1")
  let target = fused_selection_test_target()
  try
    fused_require_catalog_authority(
      admitted,
      target,
      fused_selection_test_digest("2"),
      target,
    )
  catch {
    MismatchedFusedArtifact(Manifest) => ()
    _ => fail("unexpected fused manifest replay rejection")
  } noraise {
    _ => fail("foreign fused manifest replay reached selection")
  }
  try
    fused_require_catalog_authority(
      admitted,
      target,
      admitted,
      fused_selection_test_target(minor=1),
    )
  catch {
    MismatchedFusedArtifact(Target) => ()
    _ => fail("unexpected fused cross-target rejection")
  } noraise {
    _ => fail("cross-target fused catalog reached selection")
  }
}

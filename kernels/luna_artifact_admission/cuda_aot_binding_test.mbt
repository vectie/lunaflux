///|
fn phase5_cuda_aot_lowering(
  specialization : @luna_specialization.LunaSpecializationRecord,
) -> @luna_cuda_aot.LunaCudaAotLowering raise {
  let program = phase5_tile_program()
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=2,
      max_shared_memory_bytes=4096L,
    ),
  )
  @luna_cuda_aot.lower_luna_residual_add_cuda_aot(
    program,
    planning,
    specialization.target(),
    specialization.profile_shape(),
    @catalog.AotKernelEntryPointId::from_stable_int(
      specialization.entry_point_id().as_int(),
    ),
    specialization.compiler_policy(),
    @luna_cuda_aot.LunaCudaAotLimits::new(
      max_source_bytes=4096,
      max_recipe_bytes=4096,
      max_module_bytes=4096L,
      max_evidence_bytes=4096,
    ),
  )
}

///|
fn phase5_cuda_aot_artifact(
  lowering : @luna_cuda_aot.LunaCudaAotLowering,
) -> @luna_cuda_aot.LunaCudaAotArtifact raise {
  let module_bytes = phase5_module_bytes()
  let digest = @luna_cuda_aot.LunaCudaArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
  )
  let receipt = @luna_cuda_aot.LunaCudaOfflineCompileReceipt::from_offline_builder(
    source_digest=lowering.source_digest(),
    recipe_digest=lowering.recipe_digest(),
    toolchain_digest_sha256=lowering
      .compiler_policy()
      .identity()
      .toolchain_digest_sha256(),
    driver_identity_sha256="3333333333333333333333333333333333333333333333333333333333333333",
    first_artifact_digest=digest,
    second_artifact_digest=digest,
  )
  @luna_cuda_aot.admit_luna_cuda_aot_output(
    lowering,
    module_bytes,
    module_bytes,
    receipt,
    @luna_cuda_aot.LunaCudaAotLimits::new(
      max_source_bytes=4096,
      max_recipe_bytes=4096,
      max_module_bytes=4096L,
      max_evidence_bytes=4096,
    ),
  )
}

///|
test "offline CUDA launch geometry derives from the admitted specialization artifact" {
  let specialization = phase5_specialization()
  let lowering = phase5_cuda_aot_lowering(specialization)
  let artifact = phase5_cuda_aot_artifact(lowering)
  let bound = @luna_cuda_aot.bind_luna_cuda_aot_specialization(
    lowering, artifact, specialization,
  )
  assert_eq(bound.specialization_digest(), specialization.digest())
  assert_eq(bound.operation_id(), specialization.operation_id())
  assert_eq(bound.operand_byte_count(), 32L)
  assert_eq(bound.launch_dimensions().grid_x(), 1)
  assert_eq(bound.launch_dimensions().block_x(), 256)
  let entry = bound.entry_point_input(
    @catalog.AotKernelFamilyId::from_stable_int(6),
  )
  assert_eq(entry.identity().entry_point_id().as_int(), 6)
  assert_eq(entry.identity().family().module_digest(), bound.module_digest())
  ignore(bound.module_input())
}

///|
test "offline CUDA artifact rejects a substituted specialization" {
  let specialization = phase5_specialization()
  let lowering = phase5_cuda_aot_lowering(specialization)
  let artifact = phase5_cuda_aot_artifact(lowering)
  let substituted = phase5_specialization(optimization_level=2)
  try
    @luna_cuda_aot.bind_luna_cuda_aot_specialization(
      lowering, artifact, substituted,
    )
  catch {
    Mismatch(Specialization) => ()
    _ => fail("unexpected substituted-specialization rejection")
  } noraise {
    _ => fail("substituted specialization was bound to compiler output")
  }
}

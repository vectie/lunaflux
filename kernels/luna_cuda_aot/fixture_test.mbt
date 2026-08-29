///|
fn luna_cuda_test_limits(
  max_module_bytes? : Int64 = 4096L,
) -> @luna_cuda_aot.LunaCudaAotLimits raise {
  @luna_cuda_aot.LunaCudaAotLimits::new(
    max_source_bytes=4096,
    max_recipe_bytes=4096,
    max_module_bytes~,
    max_evidence_bytes=4096,
  )
}

///|
fn luna_cuda_test_program(
  semantic? : Bool = true,
  width? : Int64 = 8L,
) -> @luna_tile_ir.LunaTileProgram raise {
  let left = @luna_tile_ir.LunaTileTensorId::from_stable_int(1)
  let right = @luna_tile_ir.LunaTileTensorId::from_stable_int(2)
  let output = @luna_tile_ir.LunaTileTensorId::from_stable_int(3)
  let tensors : Array[@luna_tile_ir.LunaTileTensorView] = []
  for id in [left, right, output] {
    tensors.push(
      @luna_tile_ir.LunaTileTensorView::new(
        id~,
        dtype=BFloat16,
        address_space=Register,
        shape=[width],
        strides_bytes=[2L],
        offset_bytes=0L,
        alignment_bytes=2L,
      ),
    )
  }
  let instructions = [
    @luna_tile_ir.LunaTileInstruction::elementwise(Add, output, [left, right]),
  ]
  let input = if semantic {
    @luna_tile_ir.LunaTileProgramInput::residual_add(
      left~,
      right~,
      output~,
      width~,
      loop_extents=[1L],
      tensors~,
      instructions~,
      constraints=[],
    )
  } else {
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors~,
      instructions~,
      constraints=[],
    )
  }
  @luna_tile_ir.admit_luna_tile_program(
    input,
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=3,
      max_instructions=1,
      max_rank=1,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=8192,
      max_shared_memory_bytes=0L,
    ),
  )
}

///|
fn luna_cuda_test_planning(
  program : @luna_tile_ir.LunaTileProgram,
) -> @luna_tile_ir.LunaTileCudaAotInput raise {
  @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=0L,
    ),
  )
}

///|
fn luna_cuda_test_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="1111111111111111111111111111111111111111111111111111111111111111",
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate~,
    max_registers=128,
  )
}

///|
fn luna_cuda_test_target(
  supports_bf16? : Bool = true,
) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16~,
    supports_cublas_lt=true,
  )
}

///|
fn luna_cuda_test_lowering() -> @luna_cuda_aot.LunaCudaAotLowering raise {
  let program = luna_cuda_test_program()
  luna_cuda_test_lowering_with(
    program,
    luna_cuda_test_planning(program),
    luna_cuda_test_target(),
    luna_cuda_test_compiler(),
  )
}

///|
fn luna_cuda_test_lowering_with(
  program : @luna_tile_ir.LunaTileProgram,
  planning : @luna_tile_ir.LunaTileCudaAotInput,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> @luna_cuda_aot.LunaCudaAotLowering raise {
  @luna_cuda_aot.lower_luna_residual_add_cuda_aot(
    program,
    planning,
    target,
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=2,
      sequence_tokens=3,
    ),
    @catalog.AotKernelEntryPointId::from_stable_int(7),
    compiler,
    luna_cuda_test_limits(),
  )
}

///|
fn luna_cuda_test_artifact_digest(
  bytes : Bytes,
) -> @luna_cuda_aot.LunaCudaArtifactDigest raise {
  @luna_cuda_aot.LunaCudaArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn luna_cuda_test_receipt(
  lowering : @luna_cuda_aot.LunaCudaAotLowering,
  first : Bytes,
  second? : Bytes = first,
) -> @luna_cuda_aot.LunaCudaOfflineCompileReceipt raise {
  @luna_cuda_aot.LunaCudaOfflineCompileReceipt::from_offline_builder(
    source_digest=lowering.source_digest(),
    recipe_digest=lowering.recipe_digest(),
    toolchain_digest_sha256=lowering
      .compiler_policy()
      .identity()
      .toolchain_digest_sha256(),
    driver_identity_sha256="3333333333333333333333333333333333333333333333333333333333333333",
    first_artifact_digest=luna_cuda_test_artifact_digest(first),
    second_artifact_digest=luna_cuda_test_artifact_digest(second),
  )
}

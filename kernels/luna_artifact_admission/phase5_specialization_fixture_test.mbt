///|
fn phase5_tile_program(
  dtype? : @luna_tile_ir.LunaTileDType = BFloat16,
  semantic? : Bool = true,
  width? : Int64 = 4L,
  elementwise? : @luna_tile_ir.LunaTileElementwiseKind = Add,
) -> @luna_tile_ir.LunaTileProgram raise {
  let first = @luna_tile_ir.LunaTileTensorId::from_stable_int(1)
  let second = @luna_tile_ir.LunaTileTensorId::from_stable_int(2)
  let output = @luna_tile_ir.LunaTileTensorId::from_stable_int(3)
  let tensors : Array[@luna_tile_ir.LunaTileTensorView] = []
  for id in [first, second, output] {
    tensors.push(
      @luna_tile_ir.LunaTileTensorView::new(
        id~,
        dtype~,
        address_space=Register,
        shape=[width],
        strides_bytes=[
          match dtype {
            BFloat16 | Float16 => 2L
            Float32 | Int32 => 4L
            Int8 => 1L
          },
        ],
        offset_bytes=0L,
        alignment_bytes=2L,
      ),
    )
  }
  let instructions = [
    @luna_tile_ir.LunaTileInstruction::elementwise(elementwise, output, [
      first, second,
    ]),
  ]
  let input = if semantic {
    @luna_tile_ir.LunaTileProgramInput::residual_add(
      left=first,
      right=second,
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
      max_rank=2,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=8192,
      max_shared_memory_bytes=0L,
    ),
  )
}

///|
fn phase5_specialization(
  optimization_level? : Int = 3,
  program_dtype? : @luna_tile_ir.LunaTileDType = BFloat16,
  program_semantic? : Bool = true,
  operation_id? : Int = 5,
  swap_plan_residual_inputs? : Bool = false,
) -> @luna_specialization.LunaSpecializationRecord raise {
  let program = phase5_tile_program(
    dtype=program_dtype,
    semantic=program_semantic,
  )
  let cuda_aot = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=2,
      max_shared_memory_bytes=4096L,
    ),
  )
  let evidence_digest = @luna_specialization.LunaSpecializationEvidenceDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  @luna_specialization.specialize_luna_tile(
    phase5_model_plan(swap_residual_inputs=swap_plan_residual_inputs),
    phase5_static_plan(),
    phase5_profile(),
    phase5_catalog(),
    operation_id=@plan.OperationId::new(operation_id),
    entry_point_id=@luna_specialization.LunaSpecializedEntryPointId::from_stable_int(
      operation_id + 1,
    ),
    program,
    cuda_aot,
    @luna_specialization.LunaCompilerPolicy::new(
      identity=@luna_specialization.LunaCompilerIdentity::new(
        toolchain_digest_sha256="1111111111111111111111111111111111111111111111111111111111111111",
        major=12,
        minor=8,
        patch=0,
      ),
      optimization_level~,
      fused_multiply_add=true,
      reassociate=false,
      max_registers=128,
    ),
    LunaBitExactOperationOrderV1,
    @luna_specialization.LunaSpecializationEvidence::new(
      scalar_referee=evidence_digest,
      admitted_tensor_rows=evidence_digest,
      dispatch_canary=evidence_digest,
      benchmark=evidence_digest,
    ),
    @luna_specialization.LunaSpecializationLimits::new(
      max_canonical_bytes=32768,
      max_bound_regions=32,
    ),
  )
}

///|
fn phase5_manifest() -> @luna_capability_manifest.LunaCapabilityManifestEntry raise {
  let specialization = phase5_specialization()
  let shape = @luna_capability_manifest.LunaGraphShapeClass::from_execution_shape(
    specialization.profile_shape(),
  )
  @luna_capability_manifest.LunaCapabilityManifestEntry::from_specialization(
    specialization~,
    architecture=@luna_capability_manifest.LunaGpuArchitectureRange::new(
      minimum_major=8,
      minimum_minor=0,
      maximum_major=9,
      maximum_minor=9,
    ),
    input_dtype=LunaBFloat16,
    accumulator_dtype=LunaAccumulatorFloat32,
    tensor_layout=LunaRowMajorStridedV1,
    kv_layout=@luna_capability_manifest.LunaKvLayoutContract::new(
      stable_version=1,
      tokens_per_page=4,
    ),
    minimum_alignment_bytes=1L,
    graph=@luna_capability_manifest.LunaGraphCapabilityMetadata::new(
      shape_class=shape,
      capture_safe=true,
      graph_memory_bytes=4096L,
      graph_memory_alignment=256L,
      eager_fallback=@luna_capability_manifest.LunaValidatedEagerFallbackDigest::from_sha256(
        "2222222222222222222222222222222222222222222222222222222222222222",
      ),
    ),
    maximum_canonical_bytes=32768,
  )
}

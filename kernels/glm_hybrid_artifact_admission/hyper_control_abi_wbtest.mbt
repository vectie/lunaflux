///|
fn hyper_control_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  sinkhorn~ : Bool,
  hostile_role? : Bool = false,
  hostile_bytes? : Bool = false,
  hostile_symbol? : Bool = false,
) -> GlmHybridArtifactClaimInput {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
  let family = @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(1),
  )
  let operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation = if sinkhorn {
    HyperConnectionSinkhorn(4, 20, 0.000001)
  } else {
    HyperConnectionFunction(4, 4096, 0.000001, 0.00001)
  }
  let capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability = if sinkhorn {
    Advanced(HyperConnectionSinkhornAot)
  } else {
    HyperConnectionFunctionAot
  }
  let operands = if sinkhorn {
    [
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(HyperInitialCombinationOutput, 2048L, 16L),
      kda_abi_operand(
        if hostile_role {
          HyperInitialCombinationOutput
        } else {
          HyperCombineControlInput
        },
        if hostile_bytes {
          2044L
        } else {
          2048L
        },
        16L,
      ),
    ]
  } else {
    [
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(HyperResidualStreamsInput, 1048576L, 16L),
      kda_abi_operand(HyperFunctionWeightsInput, 786432L, 16L),
      kda_abi_operand(HyperBaseInput, 96L, 16L),
      kda_abi_operand(HyperScaleInput, 12L, 16L),
      kda_abi_operand(HyperPostControlInput, 512L, 16L),
      kda_abi_operand(HyperInitialCombinationOutput, 2048L, 16L),
      kda_abi_operand(
        if hostile_role {
          HyperResidualStreamsInput
        } else {
          HyperCollapsedOutput
        },
        if hostile_bytes {
          262142L
        } else {
          262144L
        },
        16L,
      ),
    ]
  }
  GlmHybridArtifactClaimInput::new(
    ordinal=0,
    source_digest=requirements.digest(),
    source_identity=requirements.source_identity(),
    target=try! @catalog.DeviceTarget::new(
      compute_major=9,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
    operation~,
    capability~,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol=if hostile_symbol {
      "wrong_hyper_stage"
    } else if sinkhorn {
      "lunaflux_glm_hybrid_hc_sinkhorn_f32_v1"
    } else {
      "lunaflux_glm_hybrid_hc_function_bf16_f32_v1"
    },
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=1,
      grid_z=1,
      block_x=1,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands~,
  )
}

///|
test "hyper function and Sinkhorn admission authenticate exact staged ABI" {
  let requirements = kda_abi_requirements(
    routed_expert_count=288,
    hyper_connection=true,
  )
  assert_true(
    exact_supported_launch_abi(
      requirements,
      hyper_control_claim(requirements, sinkhorn=false),
    ),
  )
  assert_true(
    exact_supported_launch_abi(
      requirements,
      hyper_control_claim(requirements, sinkhorn=true),
    ),
  )
  for sinkhorn in [false, true] {
    assert_false(
      exact_supported_launch_abi(
        requirements,
        hyper_control_claim(requirements, sinkhorn~, hostile_role=true),
      ),
    )
    assert_false(
      exact_supported_launch_abi(
        requirements,
        hyper_control_claim(requirements, sinkhorn~, hostile_bytes=true),
      ),
    )
    assert_false(
      exact_supported_launch_abi(
        requirements,
        hyper_control_claim(requirements, sinkhorn~, hostile_symbol=true),
      ),
    )
  }
}

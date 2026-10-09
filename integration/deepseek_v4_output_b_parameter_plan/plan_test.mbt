///|
fn output_b_fixture(
  model_id : StringView,
  fill : Char,
) -> (
  @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  @advanced_decoder_cuda_aot.AdvancedDecoderOutputBProjectionCudaCandidate,
) raise {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    model_id,
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    try! @spec.ContentDigest::from_sha256(fill.to_string().repeat(64)),
  )
  let requirements = @deepseek_v4_kernel_requirements.project(
    @deepseek_v4.build(metadata),
  )
  let mut ordinal = -1
  for index, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputBProjectionV1(_, _, _) {
      ordinal = index
    }
  }
  let target = try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, 'd'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let candidate = @advanced_decoder_cuda_aot.lower_attention_output_b_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=2,
    target~,
    compiler~,
    limits=try! @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits::new(
      maximum_tokens=4096,
      maximum_source_bytes=65536,
      maximum_recipe_bytes=65536,
    ),
  )
  (metadata, requirements, candidate)
}

///|
fn output_b_evidence(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderOutputBProjectionCudaCandidate,
  contract? : @deepseek_v4_weights.DeepSeekV4OutputBQuantizationContract,
  digest? : String,
  weight_shape? : Array[Int64],
  weight_bytes? : Int64,
) -> DeepSeekV4OutputBParameterEvidenceInput {
  let layout = @deepseek_v4_weights.output_b_quantized_layout(metadata.spec())
  try! DeepSeekV4OutputBParameterEvidenceInput::new(
    contract=contract.unwrap_or(layout.contract()),
    weight_shape=weight_shape.unwrap_or([
      layout.weight_shape()[0],
      layout.weight_shape()[1],
    ]),
    scale_shape=layout.scale_shape(),
    layout_digest_sha256=digest.unwrap_or(output_b_layout_digest_sha256(layout)),
    weight_operand_ordinal=2,
    weight_operand_role=AttentionOutputBWeightCodesInput,
    weight_operand_bytes=weight_bytes.unwrap_or(
      candidate.operands()[2].byte_count(),
    ),
    scale_operand_ordinal=3,
    scale_operand_role=AttentionOutputBWeightScalesInput,
    scale_operand_bytes=candidate.operands()[3].byte_count(),
  )
}

///|
test "all five profiles bind dense Output-B FP8 independently of expert storage" {
  for
    entry in [
      ("deepseek-ai/DeepSeek-V4-Flash", '1'),
      ("deepseek-ai/DeepSeek-V4-Flash-Base", '2'),
      ("deepseek-ai/DeepSeek-V4-Pro", '3'),
      ("deepseek-ai/DeepSeek-V4-Pro-Base", '4'),
      ("deepseek-ai/DeepSeek-V4-Flash-0731", '5'),
    ] {
    let (metadata, requirements, candidate) = output_b_fixture(entry.0, entry.1)
    let plan = plan_output_b_parameter(
      metadata,
      requirements,
      candidate,
      output_b_evidence(metadata, candidate),
    )
    assert_eq(plan.status(), InertUnqualified)
    assert_eq(plan.block_width(), 128)
    assert_eq(plan.layout_digest_sha256().length(), 64)
    assert_eq(plan.digest_sha256().length(), 64)
    assert_false(candidate.owns_collective())
  }
}

///|
test "Output-B parameter join rejects shape digest bytes and cross-model identity" {
  let (metadata, requirements, candidate) = output_b_fixture(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  try
    plan_output_b_parameter(
      metadata,
      requirements,
      candidate,
      output_b_evidence(metadata, candidate, weight_shape=[4096L, 8191L]),
    )
  catch {
    ShapeMismatch => ()
    _ => fail("wrong shape failure")
  } noraise {
    _ => fail("wrong shape admitted")
  }
  try
    plan_output_b_parameter(
      metadata,
      requirements,
      candidate,
      output_b_evidence(metadata, candidate, digest=String::make(64, '0')),
    )
  catch {
    LayoutDigestMismatch => ()
    _ => fail("wrong digest failure")
  } noraise {
    _ => fail("wrong digest admitted")
  }
  try
    plan_output_b_parameter(
      metadata,
      requirements,
      candidate,
      output_b_evidence(
        metadata,
        candidate,
        weight_bytes=candidate.operands()[2].byte_count() - 1L,
      ),
    )
  catch {
    OperandMismatch => ()
    _ => fail("wrong bytes failure")
  } noraise {
    _ => fail("wrong bytes admitted")
  }
  let (other_metadata, _, _) = output_b_fixture(
    "deepseek-ai/DeepSeek-V4-Flash", 'b',
  )
  try
    plan_output_b_parameter(
      other_metadata,
      requirements,
      candidate,
      output_b_evidence(metadata, candidate),
    )
  catch {
    ModelIdentityMismatch => ()
    _ => fail("wrong identity failure")
  } noraise {
    _ => fail("cross-model identity admitted")
  }
}

///|
test "Output-B join exposes no downstream authority" {
  let (metadata, requirements, candidate) = output_b_fixture(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'c',
  )
  let plan = plan_output_b_parameter(
    metadata,
    requirements,
    candidate,
    output_b_evidence(metadata, candidate),
  )
  inspect(
    plan.digest_sha256(),
    content="614cbc83d9195f46ac9ff372188eb0486d149e1f49e12b3970b94ff78d9c98e0",
  )
  try plan.require_collective_authority() catch {
    NoCollectiveAuthority => ()
    _ => fail("wrong authority failure")
  } noraise {
    _ => fail("collective authority fabricated")
  }
  try plan.require_launch_authority() catch {
    NoLaunchAuthority => ()
    _ => fail("wrong launch failure")
  } noraise {
    _ => fail("launch authority fabricated")
  }
}

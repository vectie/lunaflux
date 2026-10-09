///|
fn parameter_fixture(
  model_id : StringView,
  fill : Char,
) -> (
  @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  @advanced_decoder_cuda_aot.AdvancedDecoderOutputAProjectionCudaCandidate,
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
    if requirement.operation() is AttentionOutputAProjectionV1(_, _, _, _) {
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
      toolchain_digest_sha256=String::make(64, 'c'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let candidate = @advanced_decoder_cuda_aot.lower_attention_output_a_projection_cuda_aot_candidate(
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
fn parameter_evidence(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderOutputAProjectionCudaCandidate,
  digest? : String,
  converted_shape? : Array[Int64],
  bytes? : Int64,
) -> DeepSeekV4OutputAParameterEvidenceInput {
  let layout = @deepseek_v4_weights.output_a_converted_layout(metadata.spec())
  let grouped = layout.converted_grouped_shape()
  try! DeepSeekV4OutputAParameterEvidenceInput::new(
    contract=layout.contract(),
    source_weight_shape=layout.source_weight_shape(),
    source_scale_shape=layout.source_scale_shape(),
    converted_grouped_shape=converted_shape.unwrap_or([
      grouped[0],
      grouped[1],
      grouped[2],
    ]),
    layout_digest_sha256=digest.unwrap_or(output_a_layout_digest_sha256(layout)),
    operand_ordinal=2,
    operand_role=AttentionOutputAWeightsInput,
    operand_byte_count=bytes.unwrap_or(candidate.operands()[2].byte_count()),
    operand_alignment=16L,
  )
}

///|
test "all profiles join converted layout to exact candidate operand 2" {
  for
    entry in [
      ("deepseek-ai/DeepSeek-V4-Flash", '1'),
      ("deepseek-ai/DeepSeek-V4-Flash-Base", '2'),
      ("deepseek-ai/DeepSeek-V4-Pro", '3'),
      ("deepseek-ai/DeepSeek-V4-Pro-Base", '4'),
      ("deepseek-ai/DeepSeek-V4-Flash-0731", '5'),
    ] {
    let (metadata, requirements, candidate) = parameter_fixture(
      entry.0,
      entry.1,
    )
    let plan = try! plan_output_a_parameter(
      metadata,
      requirements,
      candidate,
      parameter_evidence(metadata, candidate),
    )
    assert_eq(plan.status(), InertUnqualified)
    assert_eq(plan.operand_ordinal(), 2)
    assert_eq(plan.operand_role(), AttentionOutputAWeightsInput)
    assert_eq(plan.group_count(), metadata.spec().output_group_count())
    assert_eq(plan.output_rank(), 1024)
    assert_eq(plan.input_width_per_group(), 4096)
    assert_eq(
      plan.operand_byte_count(),
      metadata.spec().output_group_count().to_int64() * 1024L * 4096L * 2L,
    )
  }
}

///|
test "parameter join rejects identity shape digest bytes and Output-B" {
  let (metadata, requirements, candidate) = parameter_fixture(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  let (_, foreign_requirements, _) = parameter_fixture(
    "deepseek-ai/DeepSeek-V4-Pro", 'b',
  )
  try
    plan_output_a_parameter(
      metadata,
      foreign_requirements,
      candidate,
      parameter_evidence(metadata, candidate),
    )
  catch {
    ModelIdentityMismatch => ()
    _ => fail("wrong identity rejection")
  } noraise {
    _ => fail("foreign profile admitted")
  }
  let (_, wrong_profile_requirements, _) = parameter_fixture(
    "deepseek-ai/DeepSeek-V4-Pro", 'a',
  )
  try
    plan_output_a_parameter(
      metadata,
      wrong_profile_requirements,
      candidate,
      parameter_evidence(metadata, candidate),
    )
  catch {
    RequirementsMismatch => ()
    _ => fail("wrong profile requirements rejection")
  } noraise {
    _ => fail("wrong profile requirements admitted")
  }
  try
    plan_output_a_parameter(
      metadata,
      requirements,
      candidate,
      parameter_evidence(metadata, candidate, converted_shape=[8L, 1024L, 4095L]),
    )
  catch {
    ShapeMismatch => ()
    _ => fail("wrong shape rejection")
  } noraise {
    _ => fail("wrong shape admitted")
  }
  try
    plan_output_a_parameter(
      metadata,
      requirements,
      candidate,
      parameter_evidence(metadata, candidate, digest=String::make(64, '0')),
    )
  catch {
    LayoutDigestMismatch => ()
    _ => fail("wrong digest rejection")
  } noraise {
    _ => fail("wrong digest admitted")
  }
  try
    plan_output_a_parameter(
      metadata,
      requirements,
      candidate,
      parameter_evidence(
        metadata,
        candidate,
        bytes=candidate.operands()[2].byte_count() - 2L,
      ),
    )
  catch {
    OperandMismatch => ()
    _ => fail("wrong bytes rejection")
  } noraise {
    _ => fail("wrong bytes admitted")
  }
  assert_eq(
    requirements.requirements()[candidate.requirement_ordinal() + 1].operation(),
    AttentionOutputBProjectionV1(1024, 8, 4096),
  )
  try
    @advanced_decoder_cuda_aot.lower_attention_output_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=candidate.requirement_ordinal() + 1,
      maximum_tokens=2,
      target=candidate.target(),
      compiler=candidate.compiler_policy(),
      limits=try! @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits::new(
        maximum_tokens=4096,
        maximum_source_bytes=65536,
        maximum_recipe_bytes=65536,
      ),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong output-B rejection")
  } noraise {
    _ => fail("output-B produced output-A parameter evidence")
  }
}

///|
test "parameter plan exposes no execution authority" {
  let (metadata, requirements, candidate) = parameter_fixture(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'd',
  )
  let plan = try! plan_output_a_parameter(
    metadata,
    requirements,
    candidate,
    parameter_evidence(metadata, candidate),
  )
  plan.require_conversion_authority() catch {
    NoConversionAuthority => ()
    _ => fail("wrong conversion failure")
  }
  plan.require_materialization_authority() catch {
    NoMaterializationAuthority => ()
    _ => fail("wrong materialization failure")
  }
  plan.require_upload_authority() catch {
    NoUploadAuthority => ()
    _ => fail("wrong upload failure")
  }
  plan.require_launch_authority() catch {
    NoLaunchAuthority => ()
    _ => fail("wrong launch failure")
  }
  inspect(
    plan.digest_sha256(),
    content="d347c62f2e9c9e9aec3444865b72793d2bd4f76210d0b25473b5e024a6103f3e",
  )
}

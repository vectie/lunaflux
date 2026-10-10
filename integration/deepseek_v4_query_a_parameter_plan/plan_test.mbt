///|
fn query_a_fixture(
  model_id : StringView,
  fill : Char,
) -> (
  @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  @advanced_decoder_cuda_aot.AdvancedDecoderQueryAProjectionCudaCandidate,
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
  let target = try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, 'e'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let candidate = @advanced_decoder_cuda_aot.lower_query_a_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=1,
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
fn query_a_evidence(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderQueryAProjectionCudaCandidate,
  digest? : String,
  weight_shape? : Array[Int64],
  weight_bytes? : Int64,
) -> DeepSeekV4QueryAParameterEvidenceInput {
  let layout = @deepseek_v4_weights.query_a_quantized_layout(metadata.spec())
  try! DeepSeekV4QueryAParameterEvidenceInput::new(
    contract=layout.contract(),
    weight_shape=weight_shape.unwrap_or([
      layout.weight_shape()[0],
      layout.weight_shape()[1],
    ]),
    scale_shape=layout.scale_shape(),
    layout_digest_sha256=digest.unwrap_or(query_a_layout_digest_sha256(layout)),
    weight_operand_ordinal=2,
    weight_operand_role=QueryAWeightCodesInput,
    weight_operand_bytes=weight_bytes.unwrap_or(
      candidate.operands()[2].byte_count(),
    ),
    scale_operand_ordinal=3,
    scale_operand_role=QueryAWeightScalesInput,
    scale_operand_bytes=candidate.operands()[3].byte_count(),
  )
}

///|
test "all five profiles join replicated Query-A checkpoint storage" {
  for
    entry in [
      ("deepseek-ai/DeepSeek-V4-Flash", '1', 4096, 1024),
      ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 4096, 1024),
      ("deepseek-ai/DeepSeek-V4-Pro", '3', 7168, 1536),
      ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 7168, 1536),
      ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 4096, 1024),
    ] {
    let (metadata, requirements, candidate) = query_a_fixture(entry.0, entry.1)
    let plan = plan_query_a_parameter(
      metadata,
      requirements,
      candidate,
      query_a_evidence(metadata, candidate),
    )
    assert_eq(plan.status(), InertUnqualified)
    assert_eq(plan.requirement_ordinal(), 1)
    assert_eq(plan.input_width(), entry.2)
    assert_eq(plan.output_width(), entry.3)
    assert_eq(plan.block_width(), 128)
    assert_eq(plan.digest_sha256().length(), 64)
    if entry.0 == "deepseek-ai/DeepSeek-V4-Flash-0731" {
      inspect(
        plan.layout_digest_sha256(),
        content="ac85c652d552003dd20744e94eefb3247dcb91b751eba2376f1b7300583746cb",
      )
      inspect(
        plan.digest_sha256(),
        content="1a59a68a1da9c8bc038a328650145c8592e838682895df3ba858f441d3ba424b",
      )
    }
  }
}

///|
test "Query-A join rejects hostile shape digest bytes and identity" {
  let (metadata, requirements, candidate) = query_a_fixture(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  try
    plan_query_a_parameter(
      metadata,
      requirements,
      candidate,
      query_a_evidence(metadata, candidate, weight_shape=[1024L, 4095L]),
    )
  catch {
    ShapeMismatch => ()
    _ => fail("wrong shape failure")
  } noraise {
    _ => fail("wrong shape admitted")
  }
  try
    plan_query_a_parameter(
      metadata,
      requirements,
      candidate,
      query_a_evidence(metadata, candidate, digest=String::make(64, '0')),
    )
  catch {
    LayoutDigestMismatch => ()
    _ => fail("wrong digest failure")
  } noraise {
    _ => fail("wrong digest admitted")
  }
  try
    plan_query_a_parameter(
      metadata,
      requirements,
      candidate,
      query_a_evidence(
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
  let (other, _, _) = query_a_fixture("deepseek-ai/DeepSeek-V4-Flash", 'b')
  try
    plan_query_a_parameter(
      other,
      requirements,
      candidate,
      query_a_evidence(metadata, candidate),
    )
  catch {
    ModelIdentityMismatch => ()
    _ => fail("wrong identity failure")
  } noraise {
    _ => fail("cross-model identity admitted")
  }
}

///|
test "Query-A join remains non-authoritative" {
  let (metadata, requirements, candidate) = query_a_fixture(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'c',
  )
  let plan = plan_query_a_parameter(
    metadata,
    requirements,
    candidate,
    query_a_evidence(metadata, candidate),
  )
  try plan.require_materialization_authority() catch {
    NoMaterializationAuthority => ()
    _ => fail("wrong materialization failure")
  } noraise {
    _ => fail("materialization authority fabricated")
  }
  try plan.require_launch_authority() catch {
    NoLaunchAuthority => ()
    _ => fail("wrong launch failure")
  } noraise {
    _ => fail("launch authority fabricated")
  }
}

///|
fn shape_text(shape : ReadOnlyArray[Int64]) -> String {
  let text = StringBuilder()
  for index, value in shape {
    if index > 0 {
      text.write_char('x')
    }
    text.write_string(value.to_string())
  }
  text.to_string()
}

///|
pub fn output_b_layout_digest_sha256(
  layout : @deepseek_v4_weights.DeepSeekV4OutputBQuantizedLayout,
) -> String {
  @crypto.bytes_to_hex_string(
    @crypto.sha256(
      @utf8.encode(
        "lunaflux.deepseek-v4-output-b-quantized-layout.v1" +
        "|contract=e4m3-ue8m0-block128-row-major-v1" +
        "|weight=" +
        shape_text(layout.weight_shape()) +
        "|scale=" +
        shape_text(layout.scale_shape()) +
        "|geometry=" +
        layout.input_width().to_string() +
        ":" +
        layout.output_width().to_string() +
        ":" +
        layout.block_width().to_string() +
        "|convert-tp-shard-axis=" +
        layout.tensor_parallel_shard_axis().to_string(),
      ),
    ),
  )
}

///|
pub fn DeepSeekV4OutputBParameterEvidenceInput::new(
  contract~ : @deepseek_v4_weights.DeepSeekV4OutputBQuantizationContract,
  weight_shape~ : ArrayView[Int64],
  scale_shape~ : ArrayView[Int64],
  layout_digest_sha256~ : String,
  weight_operand_ordinal~ : Int,
  weight_operand_role~ : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  weight_operand_bytes~ : Int64,
  scale_operand_ordinal~ : Int,
  scale_operand_role~ : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  scale_operand_bytes~ : Int64,
) -> Self raise DeepSeekV4OutputBParameterPlanError {
  if weight_shape.length() != 2 ||
    scale_shape.length() != 2 ||
    layout_digest_sha256.length() != 64 ||
    weight_operand_ordinal < 0 ||
    scale_operand_ordinal < 0 ||
    weight_operand_bytes <= 0L ||
    scale_operand_bytes <= 0L {
    raise InvalidEvidence
  }
  {
    contract,
    weight_shape: ReadOnlyArray::from_array(weight_shape.to_owned()),
    scale_shape: ReadOnlyArray::from_array(scale_shape.to_owned()),
    layout_digest_sha256,
    weight_operand_ordinal,
    weight_operand_role,
    weight_operand_bytes,
    scale_operand_ordinal,
    scale_operand_role,
    scale_operand_bytes,
  }
}

///|
pub fn plan_output_b_parameter(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderOutputBProjectionCudaCandidate,
  evidence : DeepSeekV4OutputBParameterEvidenceInput,
) -> DeepSeekV4OutputBParameterPlan raise DeepSeekV4OutputBParameterPlanError {
  if requirements.source_identity().content() != metadata.content_digest() {
    raise ModelIdentityMismatch
  }
  let expected_requirements = @deepseek_v4_kernel_requirements.project(
    @deepseek_v4.build(metadata),
  ) catch {
    _ => raise RequirementsMismatch
  }
  if requirements.source_identity() != expected_requirements.source_identity() ||
    requirements.digest() != expected_requirements.digest() ||
    candidate.source_identity() != requirements.source_identity() ||
    candidate.requirements_digest() != requirements.digest() {
    raise RequirementsMismatch
  }
  let ordinal = candidate.requirement_ordinal()
  guard requirements.requirements().get(ordinal) is Some(requirement) else {
    raise UnsupportedOperation
  }
  guard requirement.operation()
    is AttentionOutputBProjectionV1(rank, groups, hidden) else {
    raise UnsupportedOperation
  }
  if requirement.capability() != AttentionOutputBProjectionV1Aot ||
    candidate.operation() != requirement.operation() ||
    candidate.capability() != requirement.capability() ||
    candidate.input_width() != rank * groups ||
    candidate.output_width() != hidden ||
    candidate.block_width() != 128 ||
    candidate.tensor_parallel_world_size() != 1 ||
    candidate.owns_collective() {
    raise CandidateMismatch
  }
  let expected = @deepseek_v4_weights.output_b_quantized_layout(metadata.spec())
  if evidence.contract != expected.contract() {
    raise QuantizationContractMismatch
  }
  if evidence.weight_shape != expected.weight_shape() ||
    evidence.scale_shape != expected.scale_shape() {
    raise ShapeMismatch
  }
  let layout_digest = output_b_layout_digest_sha256(expected)
  if evidence.layout_digest_sha256 != layout_digest {
    raise LayoutDigestMismatch
  }
  let operands = candidate.operands()
  if operands.length() != 5 ||
    evidence.weight_operand_ordinal != 2 ||
    evidence.scale_operand_ordinal != 3 ||
    evidence.weight_operand_role != AttentionOutputBWeightCodesInput ||
    evidence.scale_operand_role != AttentionOutputBWeightScalesInput ||
    operands[2].role() != evidence.weight_operand_role ||
    operands[3].role() != evidence.scale_operand_role ||
    operands[2].byte_count() != evidence.weight_operand_bytes ||
    operands[3].byte_count() != evidence.scale_operand_bytes {
    raise OperandMismatch
  }
  let digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(
      @utf8.encode(
        "lunaflux.deepseek-v4-output-b-parameter-plan.v1" +
        "|content=" +
        requirements.source_identity().content().as_hex() +
        "|plan=" +
        requirements.source_identity().plan().as_hex() +
        "|requirements=" +
        requirements.digest().as_hex() +
        "|ordinal=" +
        ordinal.to_string() +
        "|layout=" +
        layout_digest +
        "|operands=2:" +
        evidence.weight_operand_bytes.to_string() +
        ",3:" +
        evidence.scale_operand_bytes.to_string() +
        "|source=" +
        candidate.source_digest().as_hex() +
        "|recipe=" +
        candidate.recipe_digest().as_hex(),
      ),
    ),
  )
  {
    source_identity: requirements.source_identity(),
    requirement_ordinal: ordinal,
    layout_digest_sha256: layout_digest,
    weight_shape: expected.weight_shape(),
    scale_shape: expected.scale_shape(),
    input_width: candidate.input_width(),
    output_width: candidate.output_width(),
    block_width: 128,
    candidate_source_digest: candidate.source_digest().as_hex(),
    candidate_recipe_digest: candidate.recipe_digest().as_hex(),
    digest_sha256: digest,
  }
}

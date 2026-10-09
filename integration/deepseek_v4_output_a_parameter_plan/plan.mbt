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
pub fn output_a_layout_digest_sha256(
  layout : @deepseek_v4_weights.DeepSeekV4OutputAConvertedLayout,
) -> String {
  let encoded = @utf8.encode(
    "lunaflux.deepseek-v4-output-a-converted-layout.v1" +
    "|contract=fp8-e4m3-ue8m0-block128-to-bfloat16-v1" +
    "|weight=" +
    shape_text(layout.source_weight_shape()) +
    "|scale=" +
    shape_text(layout.source_scale_shape()) +
    "|converted=" +
    shape_text(layout.converted_grouped_shape()),
  )
  @crypto.bytes_to_hex_string(@crypto.sha256(encoded))
}

///|
pub fn DeepSeekV4OutputAParameterEvidenceInput::new(
  contract~ : @deepseek_v4_weights.DeepSeekV4OutputAConversionContract,
  source_weight_shape~ : ArrayView[Int64],
  source_scale_shape~ : ArrayView[Int64],
  converted_grouped_shape~ : ArrayView[Int64],
  layout_digest_sha256~ : String,
  operand_ordinal~ : Int,
  operand_role~ : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  operand_byte_count~ : Int64,
  operand_alignment~ : Int64,
) -> Self raise DeepSeekV4OutputAParameterPlanError {
  if source_weight_shape.length() != 2 ||
    source_scale_shape.length() != 2 ||
    converted_grouped_shape.length() != 3 ||
    layout_digest_sha256.length() != 64 ||
    operand_ordinal < 0 ||
    operand_byte_count <= 0L ||
    operand_alignment <= 0L {
    raise InvalidEvidence
  }
  {
    contract,
    source_weight_shape: ReadOnlyArray::from_array(
      source_weight_shape.to_owned(),
    ),
    source_scale_shape: ReadOnlyArray::from_array(source_scale_shape.to_owned()),
    converted_grouped_shape: ReadOnlyArray::from_array(
      converted_grouped_shape.to_owned(),
    ),
    layout_digest_sha256,
    operand_ordinal,
    operand_role,
    operand_byte_count,
    operand_alignment,
  }
}

///|
pub fn plan_output_a_parameter(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderOutputAProjectionCudaCandidate,
  evidence : DeepSeekV4OutputAParameterEvidenceInput,
) -> DeepSeekV4OutputAParameterPlan raise DeepSeekV4OutputAParameterPlanError {
  if requirements.source_identity().content() != metadata.content_digest() {
    raise ModelIdentityMismatch
  }
  let expected_requirements = @deepseek_v4_kernel_requirements.project(
    @deepseek_v4.build(metadata),
  ) catch {
    _ => raise RequirementsMismatch
  }
  if requirements.source_identity() != expected_requirements.source_identity() ||
    requirements.digest() != expected_requirements.digest() {
    raise RequirementsMismatch
  }
  if candidate.source_identity() != requirements.source_identity() ||
    candidate.requirements_digest() != requirements.digest() {
    raise RequirementsMismatch
  }
  let ordinal = candidate.requirement_ordinal()
  guard requirements.requirements().get(ordinal) is Some(requirement) else {
    raise UnsupportedOperation
  }
  guard requirement.operation()
    is AttentionOutputAProjectionV1(rank, groups, heads, head_dimension) else {
    raise UnsupportedOperation
  }
  if requirement.capability() != AttentionOutputAProjectionV1Aot ||
    candidate.operation() != requirement.operation() ||
    candidate.capability() != requirement.capability() ||
    candidate.group_count() != groups ||
    candidate.output_rank() != rank ||
    heads % groups != 0 ||
    candidate.input_width_per_group() != heads / groups * head_dimension {
    raise CandidateMismatch
  }
  let expected = @deepseek_v4_weights.output_a_converted_layout(metadata.spec())
  if evidence.contract != expected.contract() {
    raise ConversionContractMismatch
  }
  if evidence.source_weight_shape != expected.source_weight_shape() ||
    evidence.source_scale_shape != expected.source_scale_shape() ||
    evidence.converted_grouped_shape != expected.converted_grouped_shape() {
    raise ShapeMismatch
  }
  let layout_digest = output_a_layout_digest_sha256(expected)
  if evidence.layout_digest_sha256 != layout_digest {
    raise LayoutDigestMismatch
  }
  let operands = candidate.operands()
  if evidence.operand_ordinal != 2 ||
    operands.length() != 4 ||
    evidence.operand_role != AttentionOutputAWeightsInput ||
    operands[2].role() != evidence.operand_role ||
    operands[2].byte_count() != evidence.operand_byte_count ||
    operands[2].alignment() != evidence.operand_alignment {
    raise OperandMismatch
  }
  let digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(
      @utf8.encode(
        "lunaflux.deepseek-v4-output-a-parameter-plan.v1" +
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
        "|operand=2:attention-output-a-weights:" +
        evidence.operand_byte_count.to_string() +
        ":" +
        evidence.operand_alignment.to_string() +
        "|geometry=" +
        groups.to_string() +
        ":" +
        rank.to_string() +
        ":" +
        candidate.input_width_per_group().to_string() +
        "|source=" +
        candidate.source_digest().as_hex() +
        "|recipe=" +
        candidate.recipe_digest().as_hex(),
      ),
    ),
  )
  {
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal: ordinal,
    contract: expected.contract(),
    source_weight_shape: expected.source_weight_shape(),
    source_scale_shape: expected.source_scale_shape(),
    converted_grouped_shape: expected.converted_grouped_shape(),
    layout_digest_sha256: layout_digest,
    operand_ordinal: 2,
    operand_role: operands[2].role(),
    operand_byte_count: operands[2].byte_count(),
    operand_alignment: operands[2].alignment(),
    group_count: groups,
    output_rank: rank,
    input_width_per_group: candidate.input_width_per_group(),
    candidate_source_digest: candidate.source_digest().as_hex(),
    candidate_recipe_digest: candidate.recipe_digest().as_hex(),
    digest_sha256: digest,
  }
}

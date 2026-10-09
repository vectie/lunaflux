///|
let token_hash_operand_ordinal : Int = 2

///|
let token_hash_table_bytes : Int64 = 129280L * 6L * 4L

///|
fn[T] from_token_hash(
  body : () -> T raise @deepseek_v4_host_materialization.DeepSeekV4TokenHashMaterializationError,
) -> T raise DeepSeekV4TokenHashOperandPlanError {
  body() catch {
    error => raise TokenHash(error)
  }
}

///|
fn source_evidence(
  tables : @deepseek_v4_host_materialization.DeepSeekV4TokenHashTables,
) -> TokenHashSourceEvidence raise DeepSeekV4TokenHashOperandPlanError {
  if tables.is_released() {
    raise SourceReleased
  }
  let entries : Array[TokenHashTableEvidence] = []
  for ordinal in 0..<tables.table_count() {
    entries.push({
      layer_index: from_token_hash(() => tables.layer_index(ordinal)),
      digest: from_token_hash(() => tables.table_digest(ordinal)).as_hex(),
      byte_count: from_token_hash(() => tables.table_byte_count(ordinal)),
    })
  }
  {
    identity: tables.identity(),
    source_manifest_digest: tables.source_manifest_digest().as_hex(),
    tables: ReadOnlyArray::from_array(entries),
    released: false,
  }
}

///|
fn candidate_evidence(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
) -> TokenHashCandidateEvidence raise DeepSeekV4TokenHashOperandPlanError {
  let operands = candidate.operands()
  guard operands.get(token_hash_operand_ordinal) is Some(operand) else {
    raise OperandOrdinalMismatch
  }
  {
    source_identity: candidate.source_identity(),
    requirements_digest: candidate.requirements_digest(),
    requirement_ordinal: candidate.requirement_ordinal(),
    operation: candidate.operation(),
    capability: candidate.capability(),
    family: candidate.family(),
    numerical_contract: candidate.numerical_contract(),
    routed_expert_count: candidate.routed_expert_count(),
    operand_count: operands.length(),
    operand_role: operand.role(),
    operand_byte_count: operand.byte_count(),
    operand_alignment: operand.alignment(),
    source_digest: candidate.source_digest(),
    recipe_digest: candidate.recipe_digest(),
    manifest_bindable: candidate.manifest_bindable(),
  }
}

///|
fn valid_digest_text(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
fn validate_source_evidence(
  source : TokenHashSourceEvidence,
) -> Unit raise DeepSeekV4TokenHashOperandPlanError {
  if source.released {
    raise SourceReleased
  }
  if source.tables.length() != 3 ||
    !valid_digest_text(source.source_manifest_digest) {
    raise InvalidSourceMetadata
  }
  for ordinal, table in source.tables {
    if table.layer_index != ordinal ||
      table.byte_count.to_int64() != token_hash_table_bytes ||
      !valid_digest_text(table.digest) {
      raise InvalidSourceMetadata
    }
  }
}

///|
fn validate_sidecar_layout(
  source : TokenHashSourceEvidence,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise DeepSeekV4TokenHashOperandPlanError {
  if layout.source_identity() != source.identity ||
    layout.source_arena_count() != 3 ||
    layout.region_count() != 3 {
    raise LayoutRegionMismatch(-1)
  }
  for ordinal, table in source.tables {
    if layout.source_arena_length(ordinal) != Some(table.byte_count) {
      raise LayoutRegionMismatch(ordinal)
    }
    guard layout.region(ordinal) is Some(region) else {
      raise LayoutRegionMismatch(ordinal)
    }
    if region.tensor_ordinal() != table.layer_index ||
      region.source_arena_ordinal() != ordinal ||
      region.source_offset() != 0 ||
      region.byte_count() != token_hash_table_bytes ||
      region.device_offset() < 0L ||
      region.device_offset() % layout.alignment_bytes() != 0L {
      raise LayoutRegionMismatch(ordinal)
    }
  }
}

///|
fn plan_from_evidence(
  source : TokenHashSourceEvidence,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  candidate : TokenHashCandidateEvidence,
  layer_index : Int,
) -> DeepSeekV4TokenHashOperandPlan raise DeepSeekV4TokenHashOperandPlanError {
  validate_source_evidence(source)
  if layer_index < 0 || layer_index >= 3 {
    raise InvalidLayer(layer_index)
  }
  if source.identity != requirements.source_identity() {
    raise SourceIdentityMismatch
  }
  validate_sidecar_layout(source, layout)
  if candidate.source_identity != requirements.source_identity() {
    raise RequirementsIdentityMismatch
  }
  if candidate.requirements_digest != requirements.digest() {
    raise RequirementsDigestMismatch
  }
  guard requirements.requirements().get(candidate.requirement_ordinal)
    is Some(requirement) else {
    raise CandidateRequirementOrdinalMismatch(candidate.requirement_ordinal)
  }
  guard requirement.operation()
    is TokenHashRouting(3, 129280, 6, routed_expert_count) else {
    raise CandidateRequirementOrdinalMismatch(candidate.requirement_ordinal)
  }
  if requirement.capability() != TokenHashRoutingAot ||
    candidate.capability != TokenHashRoutingAot {
    raise CandidateCapabilityMismatch
  }
  if candidate.operation != requirement.operation() ||
    candidate.routed_expert_count != routed_expert_count {
    raise CandidateOperationMismatch
  }
  if candidate.family != TokenHashRoutingI32 ||
    candidate.numerical_contract != BitExactInt32TableLookupV1 {
    raise CandidateContractMismatch
  }
  if candidate.manifest_bindable {
    raise CandidateUnexpectedlyBindable
  }
  if candidate.operand_count != 4 {
    raise OperandOrdinalMismatch
  }
  if candidate.operand_role != TokenHashTableInput {
    raise OperandRoleMismatch
  }
  if candidate.operand_byte_count != token_hash_table_bytes {
    raise OperandByteCountMismatch
  }
  if candidate.operand_alignment != 4L {
    raise OperandAlignmentMismatch
  }
  guard source.tables.get(layer_index) is Some(table) else {
    raise InvalidLayer(layer_index)
  }
  guard layout.region(layer_index) is Some(region) else {
    raise LayoutRegionMismatch(layer_index)
  }
  let without_digest : DeepSeekV4TokenHashOperandPlan = {
    source_identity: source.identity,
    requirements_digest: requirements.digest(),
    candidate_source_digest: candidate.source_digest,
    candidate_recipe_digest: candidate.recipe_digest,
    sidecar_layout_digest: layout.digest(),
    sidecar_source_digest: layout.source_digest(),
    source_manifest_digest: source.source_manifest_digest,
    table_digest: table.digest,
    requirement_ordinal: candidate.requirement_ordinal,
    layer_index,
    routed_expert_count,
    operand_ordinal: token_hash_operand_ordinal,
    operand_role: candidate.operand_role,
    byte_count: candidate.operand_byte_count,
    alignment: candidate.operand_alignment,
    region_ordinal: layer_index,
    region_tensor_ordinal: region.tensor_ordinal(),
    device_offset: region.device_offset(),
    digest: DeepSeekV4TokenHashOperandPlanDigest(String::make(64, '0')),
  }
  { ..without_digest, digest: canonical_operand_plan_digest(without_digest), }
}

///|
/// Joins one live official DeepSeek hash table, its validated isolated device
/// region, and the exact non-bindable I32 token-hash CUDA source candidate.
pub fn plan_token_hash_table_operand(
  tables : @deepseek_v4_host_materialization.DeepSeekV4TokenHashTables,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
  layer_index~ : Int,
) -> DeepSeekV4TokenHashOperandPlan raise DeepSeekV4TokenHashOperandPlanError {
  @deepseek_v4_device_upload.validate_token_hash_sidecar_source(tables, layout) catch {
    error => raise Sidecar(error)
  }
  plan_from_evidence(
    source_evidence(tables),
    layout,
    requirements,
    candidate_evidence(candidate),
    layer_index,
  )
}

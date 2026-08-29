///|
fn readonly_document_value(
  bytes : Bytes,
  key : String,
  rule : ReadOnlyProductionRule,
) -> String raise ReadOnlyProductionError {
  let document = @utf8.decode(bytes) catch {
    _ => raise InvalidReadOnlyProduction(rule)
  }
  let prefix = key + "="
  let mut found : String? = None
  for line in document.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) || line.length() == prefix.length() {
        raise InvalidReadOnlyProduction(rule)
      }
      found = Some(line[prefix.length():].to_owned())
    }
  }
  guard found is Some(value) else { raise InvalidReadOnlyProduction(rule) }
  value
}

///|
fn readonly_exact_int(
  text : String,
  rule : ReadOnlyProductionRule,
) -> Int raise ReadOnlyProductionError {
  let value : Int = @strconv.from_str(text) catch {
    _ => raise InvalidReadOnlyProduction(rule)
  }
  if value.to_string() != text {
    raise InvalidReadOnlyProduction(rule)
  }
  value
}

///|
fn readonly_exact_int64(
  text : String,
  rule : ReadOnlyProductionRule,
) -> Int64 raise ReadOnlyProductionError {
  let value : Int64 = @strconv.from_str(text) catch {
    _ => raise InvalidReadOnlyProduction(rule)
  }
  if value.to_string() != text {
    raise InvalidReadOnlyProduction(rule)
  }
  value
}

///|
fn extract_readonly_qualified_identity(
  qualified : @luna_paged_attention_readonly_physical_evidence.QualifiedPagedAttentionReadOnlyArtifact,
  policy : @luna_paged_attention_readonly_physical_evidence.ApprovedPagedAttentionReadOnlyPhysicalPolicy,
) -> ReadOnlyQualifiedIdentity raise ReadOnlyProductionError {
  if !qualified.physically_qualified() ||
    qualified.manifest_bindable() ||
    qualified.promotion_authorized() {
    raise MismatchedReadOnlyProduction(Qualification)
  }
  let candidate = qualified.candidate()
  let compiled = qualified.compiled()
  let evidence = qualified.evidence()
  if candidate.manifest_bindable() ||
    candidate.promotion_authorized() ||
    compiled.manifest_bindable() ||
    compiled.promotion_authorized() ||
    evidence.manifest_bindable() ||
    evidence.promotion_authorized() {
    raise MismatchedReadOnlyProduction(Qualification)
  }
  if candidate.candidate_digest() != compiled.candidate_digest() ||
    candidate.source_digest() != compiled.source_digest() ||
    candidate.recipe_digest() != compiled.recipe_digest() ||
    candidate.candidate_digest().as_hex() != evidence.candidate_digest() ||
    compiled.binding_digest().as_hex() != evidence.compiled_binding_digest() ||
    compiled.module_digest().as_hex() != evidence.cubin_digest().as_hex() {
    raise MismatchedReadOnlyProduction(CompiledArtifact)
  }
  if evidence.operation_id() != candidate.operation_id() ||
    evidence.input_activation() != candidate.input_activation() ||
    evidence.input_row_width() != candidate.input_row_width() ||
    evidence.dispatch_canary_per_token() !=
    candidate.dispatch_canary_per_token() ||
    evidence.dispatch_canary_cell_count() !=
    candidate.profile().max_query_tokens() {
    raise MismatchedReadOnlyProduction(PhysicalEvidence)
  }
  let physical = evidence.canonical_bytes()
  let policy_bytes = policy.canonical_bytes()
  let policy_digest = policy.digest().as_hex()
  if readonly_document_value(physical, "approved_policy_sha256", PhysicalPolicy) !=
    policy_digest {
    raise MismatchedReadOnlyProduction(PhysicalPolicy)
  }
  let driver = readonly_document_value(
    physical,
    "driver_identity_sha256",
    PhysicalPolicy,
  )
  let policy_driver = readonly_document_value(
    policy_bytes,
    "driver_identity_sha256",
    PhysicalPolicy,
  )
  let device_uuid = readonly_document_value(
    policy_bytes,
    "device_uuid",
    PhysicalPolicy,
  )
  let device_pci_bus_id = readonly_document_value(
    policy_bytes,
    "device_pci_bus_id",
    PhysicalPolicy,
  )
  let device_name_hex = readonly_document_value(
    policy_bytes,
    "device_name_hex",
    PhysicalPolicy,
  )
  let device_total_memory_bytes = readonly_document_value(
    policy_bytes,
    "device_total_memory_bytes",
    PhysicalPolicy,
  )
  let physical_operation = readonly_exact_int(
    readonly_document_value(physical, "operation_id", Operation),
    Operation,
  )
  let physical_activation = readonly_exact_int(
    readonly_document_value(physical, "input_activation_ref", Activation),
    Activation,
  )
  let physical_row_width = readonly_exact_int(
    readonly_document_value(physical, "input_row_width", Activation),
    Activation,
  )
  let physical_memory = readonly_exact_int64(
    device_total_memory_bytes,
    PhysicalPolicy,
  )
  if physical_operation != candidate.operation_id().as_int() ||
    physical_activation != candidate.input_activation().as_int() ||
    physical_row_width != candidate.input_row_width() ||
    physical_memory <= 0L {
    raise MismatchedReadOnlyProduction(PhysicalEvidence)
  }
  if driver != policy_driver ||
    readonly_document_value(physical, "device_uuid", PhysicalPolicy) !=
    device_uuid ||
    readonly_document_value(physical, "device_pci_bus_id", PhysicalPolicy) !=
    device_pci_bus_id ||
    readonly_document_value(physical, "device_name_hex", PhysicalPolicy) !=
    device_name_hex ||
    readonly_document_value(
      physical,
      "device_total_memory_bytes",
      PhysicalPolicy,
    ) !=
    device_total_memory_bytes {
    raise MismatchedReadOnlyProduction(PhysicalPolicy)
  }
  let reference = candidate.reference_fallback()
  let target = candidate.target()
  let profile = candidate.profile()
  if target.compute_major() != 12 ||
    target.compute_minor() != 0 ||
    !target.supports_bf16() ||
    readonly_document_value(policy_bytes, "target", Target) != "sm_120" {
    raise MismatchedReadOnlyProduction(Target)
  }
  let canary = candidate.dispatch_canary_per_token()
  if canary <= 0 ||
    canary.to_int64() * profile.max_query_tokens().to_int64() > 4294967295L {
    raise InvalidReadOnlyProduction(Canary)
  }
  {
    candidate: candidate.candidate_digest().as_hex(),
    source: candidate.source_digest().as_hex(),
    recipe: candidate.recipe_digest().as_hex(),
    fallback_source: reference.source_digest().as_hex(),
    fallback_recipe: reference.recipe_digest().as_hex(),
    compiled_binding: compiled.binding_digest().as_hex(),
    module_digest: compiled.module_digest().as_hex(),
    compile_receipt: compiled.compile_receipt_digest().as_hex(),
    physical_evidence: evidence.digest().as_hex(),
    physical_files_manifest: evidence.files_manifest_digest().as_hex(),
    physical_outer_seal: evidence.outer_seal_digest().as_hex(),
    approved_physical_policy: policy_digest,
    toolchain: reference.compiler_policy().identity().toolchain_digest_sha256(),
    driver,
    device_uuid,
    device_pci_bus_id,
    device_name_hex,
    device_total_memory_bytes,
    operation: candidate.operation_id().as_int(),
    activation: candidate.input_activation().as_int(),
    input_row_width: candidate.input_row_width(),
    profile_id: profile.id().as_int(),
    profile_rows: profile.max_query_rows(),
    profile_tokens: profile.max_query_tokens(),
    profile_page_entries: profile.max_page_table_entries(),
    tokens_per_page: reference.tokens_per_page(),
    page_stride_bytes: reference.page_stride_bytes(),
    physical_page_capacity: reference.physical_page_capacity(),
    dispatch_canary: canary,
    dispatch_canary_cells: profile.max_query_tokens(),
    symbol: candidate.function_symbol(),
    abi: candidate.kernel_abi().canonical_name(),
    target,
  }
}

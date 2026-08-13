///|
fn parse_manifest_text(
  document : StringView,
  evidence : ManifestContractEvidence,
  limits : KernelArtifactFileLimits,
) -> KernelArtifactManifest raise KernelArtifactFileError {
  @json_guard.reject_duplicate_object_keys(
    document,
    max_depth=limits.max_json_depth(),
  ) catch {
    InvalidDepthLimit => raise MalformedJson
    DepthLimitExceeded(_) =>
      raise JsonDepthExceeded(maximum=limits.max_json_depth())
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise MalformedJson }
  validate_fields(root, ManifestRoot, [
    ("schema_version", SchemaVersionField),
    ("model_identity", ModelIdentityField),
    ("device_target", DeviceTargetField),
    ("catalog_version", CatalogVersionField),
    ("modules", ModulesField),
  ])
  if require_nonnegative_int(
      root,
      "schema_version",
      ManifestRoot,
      SchemaVersionField,
    ) !=
    1 {
    raise invalid_field(ManifestRoot, SchemaVersionField, UnsupportedValue)
  }
  validate_identity(root, evidence.model_identity)
  validate_device_target(root, evidence.device_target)
  validate_catalog_version(root, evidence.catalog_version)
  parse_modules(root, limits)
}

///|
fn read_manifest_with_evidence(
  document : BytesView,
  evidence : ManifestContractEvidence,
  limits : KernelArtifactFileLimits,
) -> KernelArtifactManifest raise KernelArtifactFileError {
  if document.length().to_int64() > limits.max_manifest_bytes() {
    raise FileTooLarge(file=ManifestFile, maximum=limits.max_manifest_bytes())
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_manifest_text(text, evidence, limits)
}

///|
/// Parses an exact v1 inert AOT manifest and pins it to the supplied admitted
/// launch contracts.
///
/// The byte bound is enforced before UTF-8 or JSON allocation. Duplicate and
/// unknown fields are rejected at every object. The result carries only
/// content identities, lexical path identifiers, stable integer entry-point
/// identities, and bounded symbol declarations; it cannot request execution,
/// discovery, compilation, environment lookup, caching, or fallback.
pub fn read_manifest(
  document : BytesView,
  contracts : @launch_contract.LaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> KernelArtifactManifest raise KernelArtifactFileError {
  read_manifest_with_evidence(
    document,
    stateless_contract_evidence(contracts),
    limits,
  )
}

///|
/// Parses an exact inert AOT manifest and pins it to the supplied paged launch
/// contracts. Parsing, bounds, duplicate-field rejection, and identity checks
/// are identical to `read_manifest`.
pub fn read_paged_manifest(
  document : BytesView,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> KernelArtifactManifest raise KernelArtifactFileError {
  read_manifest_with_evidence(
    document,
    paged_contract_evidence(contracts),
    limits,
  )
}

///|
fn parse_entry_point(
  value : Json,
  digest : @catalog.AotArtifactDigest,
  index : Int,
  maximum_symbol_bytes : Int,
) -> @artifact.KernelEntryPointInput raise KernelArtifactFileError {
  guard value is Object(entry) else {
    raise invalid_field(
      ManifestModule,
      EntryPointsField,
      ExpectedObject,
      index~,
    )
  }
  validate_fields(entry, ManifestEntryPoint, [
    ("family_id", FamilyIdField),
    ("entry_point_id", EntryPointIdField),
    ("symbol", FunctionSymbolField),
  ])
  let family_value = require_nonnegative_int(
    entry,
    "family_id",
    ManifestEntryPoint,
    FamilyIdField,
    index~,
  )
  let entry_value = require_nonnegative_int(
    entry,
    "entry_point_id",
    ManifestEntryPoint,
    EntryPointIdField,
    index~,
  )
  let family_id = @catalog.AotKernelFamilyId::from_stable_int(family_value) catch {
    _ =>
      raise invalid_field(
        ManifestEntryPoint,
        FamilyIdField,
        UnsupportedValue,
        index~,
      )
  }
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(
    entry_value,
  ) catch {
    _ =>
      raise invalid_field(
        ManifestEntryPoint,
        EntryPointIdField,
        UnsupportedValue,
        index~,
      )
  }
  let symbol = require_string(
    entry,
    "symbol",
    ManifestEntryPoint,
    FunctionSymbolField,
  )
  if symbol.length() > maximum_symbol_bytes {
    raise invalid_field(
      ManifestEntryPoint,
      FunctionSymbolField,
      StringTooLong,
      index~,
    )
  }
  @artifact.KernelEntryPointInput::new(
    entry_point=@catalog.AotKernelEntryPoint::new(
      family=@catalog.AotKernelFamily::new(module_digest=digest, family_id~),
      entry_point_id~,
    ),
    function_symbol=symbol,
  )
}

///|
fn parse_modules(
  root : Map[String, Json],
  limits : KernelArtifactFileLimits,
) -> KernelArtifactManifest raise KernelArtifactFileError {
  let module_values = require_array(root, "modules", ManifestRoot, ModulesField)
  let artifact_limits = limits.artifact()
  if module_values.length() > artifact_limits.max_modules() {
    raise invalid_field(ManifestRoot, ModulesField, CountExceeded)
  }
  let digests : Array[@catalog.AotArtifactDigest] = []
  let paths : Array[String] = []
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for module_index, value in module_values {
    guard value is Object(kernel_module) else {
      raise invalid_field(
        ManifestRoot,
        ModulesField,
        ExpectedObject,
        index=module_index,
      )
    }
    validate_fields(kernel_module, ManifestModule, [
      ("sha256", ModuleSha256Field),
      ("path", ModulePathField),
      ("entry_points", EntryPointsField),
    ])
    let digest_text = require_string(
      kernel_module,
      "sha256",
      ManifestModule,
      ModuleSha256Field,
    )
    let digest = @catalog.AotArtifactDigest::from_sha256(digest_text) catch {
      _ =>
        raise invalid_field(
          ManifestModule,
          ModuleSha256Field,
          InvalidSha256,
          index=module_index,
        )
    }
    let path = require_string(
      kernel_module,
      "path",
      ManifestModule,
      ModulePathField,
    )
    validate_path_identifier(
      path,
      ModuleFile(module_index),
      limits.max_path_bytes(),
    )
    if paths.contains(path) {
      raise InvalidPath(file=ModuleFile(module_index), issue=DuplicatePath)
    }
    let entry_values = require_array(
      kernel_module,
      "entry_points",
      ManifestModule,
      EntryPointsField,
    )
    if entry_values.length() >
      artifact_limits.max_entry_points() - entries.length() {
      raise invalid_field(
        ManifestModule,
        EntryPointsField,
        CountExceeded,
        index=module_index,
      )
    }
    for entry_index, entry_value in entry_values {
      entries.push(
        parse_entry_point(
          entry_value,
          digest,
          entry_index,
          artifact_limits.max_function_symbol_bytes(),
        ),
      )
    }
    digests.push(digest)
    paths.push(path)
  }
  {
    module_digests: ReadOnlyArray::from_array(digests),
    module_path_identifiers: ReadOnlyArray::from_array(paths),
    entry_points: ReadOnlyArray::from_array(entries),
  }
}

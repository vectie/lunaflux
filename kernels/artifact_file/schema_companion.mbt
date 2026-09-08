///|
/// Optional exact launch for the ordered companion entry. Absence preserves
/// the legacy primary launch; a partial declaration is never defaulted.
pub fn decode_companion_dimensions(
  value : Json,
) -> @launch_contract.AotLaunchDimensions raise KernelArtifactFileError {
  guard value is Object(fields) else {
    raise invalid_field(ManifestEntryPoint, EntryPointsField, ExpectedObject)
  }
  validate_fields(fields, ManifestEntryPoint, [
    ("grid_x", EntryPointsField),
    ("grid_y", EntryPointsField),
    ("grid_z", EntryPointsField),
    ("block_x", EntryPointsField),
    ("block_y", EntryPointsField),
    ("block_z", EntryPointsField),
    ("shared_memory_bytes", EntryPointsField),
  ])
  let integer = fn(name : String) -> Int raise KernelArtifactFileError {
    require_nonnegative_int(fields, name, ManifestEntryPoint, EntryPointsField)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=integer("grid_x"),
    grid_y=integer("grid_y"),
    grid_z=integer("grid_z"),
    block_x=integer("block_x"),
    block_y=integer("block_y"),
    block_z=integer("block_z"),
    shared_memory_bytes=integer("shared_memory_bytes"),
  ) catch {
    _ =>
      raise invalid_field(
        ManifestEntryPoint,
        EntryPointsField,
        UnsupportedValue,
      )
  }
}

///|
fn parse_companion_dimensions(
  fields : Map[String, Json],
) -> @launch_contract.AotLaunchDimensions? raise KernelArtifactFileError {
  match fields.get("companion_dimensions") {
    None => None
    Some(value) => Some(decode_companion_dimensions(value))
  }
}

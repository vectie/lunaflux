///|
/// Decode the bounded optional entry-point extension shared by standalone and
/// full-graph artifact manifests. Semantic symbol checks run during admission.
pub fn decode_row_variants(
  value : Json,
  maximum_symbol_bytes : Int,
) -> ReadOnlyArray[@artifact.KernelRowVariant] raise KernelArtifactFileError {
  ReadOnlyArray::from_array(
    parse_row_variants({ "row_variants": value }, maximum_symbol_bytes),
  )
}

///|
fn parse_row_variants(
  entry : Map[String, Json],
  maximum_symbol_bytes : Int,
) -> Array[@artifact.KernelRowVariant] raise KernelArtifactFileError {
  guard entry.get("row_variants") is Some(value) else { return [] }
  guard value is Array(values) && values.length() <= 19 else {
    raise invalid_field(ManifestEntryPoint, EntryPointsField, CountExceeded)
  }
  let variants : Array[@artifact.KernelRowVariant] = []
  for index, value in values {
    guard value is Object(fields) else {
      raise invalid_field(
        ManifestEntryPoint,
        EntryPointsField,
        ExpectedObject,
        index~,
      )
    }
    validate_fields(fields, ManifestEntryPoint, [
      ("maximum_token_rows", EntryPointsField),
      ("symbol", FunctionSymbolField),
      ("grid_x", EntryPointsField),
      ("block_x", EntryPointsField),
      ("shared_memory_bytes", EntryPointsField),
    ])
    let symbol = require_string(
      fields,
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
    let integer = fn(name : String) -> Int raise KernelArtifactFileError {
      require_nonnegative_int(
        fields,
        name,
        ManifestEntryPoint,
        EntryPointsField,
        index~,
      )
    }
    let dimensions = @launch_contract.AotLaunchDimensions::new(
      grid_x=integer("grid_x"),
      grid_y=1,
      grid_z=1,
      block_x=integer("block_x"),
      block_y=1,
      block_z=1,
      shared_memory_bytes=integer("shared_memory_bytes"),
    ) catch {
      _ =>
        raise invalid_field(
          ManifestEntryPoint,
          EntryPointsField,
          UnsupportedValue,
          index~,
        )
    }
    variants.push(
      @artifact.KernelRowVariant::new(
        maximum_token_rows=integer("maximum_token_rows"),
        function_symbol=symbol,
        dimensions~,
      ),
    )
  }
  variants
}

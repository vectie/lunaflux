///|
fn require_recipe_capacity(
  output : Array[Byte],
  additional : Int,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeError {
  if additional < 0 || output.length() > maximum - additional {
    raise InvalidFp8RuntimeRecipe(CanonicalBytes)
  }
}

///|
fn append_recipe_text(
  output : Array[Byte],
  value : String,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeError {
  let bytes = @utf8.encode(value)
  require_recipe_capacity(output, 8, maximum)
  for byte in bytes.length().to_int64().reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
  require_recipe_capacity(output, bytes.length(), maximum)
  for byte in bytes {
    output.push(byte)
  }
}

///|
fn append_recipe_region(
  output : Array[Byte],
  region : @fp8_launch_abi.Fp8LaunchStorageRegion,
  maximum : Int,
) -> Unit raise Fp8RuntimeRecipeError {
  append_recipe_text(output, region.storage().as_sha256(), maximum)
  append_recipe_text(output, region.storage_bytes().to_string(), maximum)
  append_recipe_text(output, region.offset().to_string(), maximum)
  append_recipe_text(output, region.byte_length().to_string(), maximum)
  append_recipe_text(output, region.alignment().to_string(), maximum)
}

///|
fn encode_runtime_recipe(
  startup : @fp8_startup_admission.Fp8StartupAdmission,
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launches : ReadOnlyArray[@fp8_launch_abi.Fp8DynamicLaunchAbi],
  weight_bindings : ReadOnlyArray[Fp8LaunchWeightBinding],
  weight_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  workspace_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  workspace_bytes : Int64,
  maximum : Int,
) -> Bytes raise Fp8RuntimeRecipeError {
  let output : Array[Byte] = []
  append_recipe_text(output, "lunaflux.fp8-runtime-recipe.v1", maximum)
  append_recipe_text(output, startup.digest().as_hex(), maximum)
  append_recipe_text(output, weights.artifact_digest().as_hex(), maximum)
  append_recipe_text(output, weight_storage.as_sha256(), maximum)
  append_recipe_text(
    output,
    weights.layout().arena_bytes().to_string(),
    maximum,
  )
  append_recipe_text(
    output,
    weights.layout().alignment_bytes().to_string(),
    maximum,
  )
  append_recipe_text(output, weights.layout().length().to_string(), maximum)
  for index in 0..<weights.layout().length() {
    let reference = try! @plan.TensorRef::new(index)
    guard weights.layout().region(reference) is Some(region) else {
      raise InvalidFp8RuntimeRecipe(WeightLayout)
    }
    append_recipe_text(output, reference.as_int().to_string(), maximum)
    append_recipe_text(output, region.offset_bytes().to_string(), maximum)
    append_recipe_text(output, region.byte_length().to_string(), maximum)
  }
  append_recipe_text(output, workspace_storage.as_sha256(), maximum)
  append_recipe_text(output, workspace_bytes.to_string(), maximum)
  append_recipe_text(output, launches.length().to_string(), maximum)
  for index, launch in launches {
    append_recipe_text(output, launch.digest().as_sha256(), maximum)
    let binding = weight_bindings[index]
    append_recipe_text(output, binding.profile().as_int().to_string(), maximum)
    append_recipe_text(
      output,
      binding.operation().as_int().to_string(),
      maximum,
    )
    for region in binding.regions() {
      append_recipe_text(output, region.tensor().as_int().to_string(), maximum)
      append_recipe_text(output, region.storage_digest().as_hex(), maximum)
      append_recipe_region(output, region.region(), maximum)
    }
    let scales = launch.scale_tensor_regions()
    append_recipe_text(output, scales.length().to_string(), maximum)
    for scale in scales {
      append_recipe_text(
        output,
        scale.tensor().tensor().as_int().to_string(),
        maximum,
      )
      append_recipe_text(output, scale.storage_digest().as_hex(), maximum)
      append_recipe_region(output, scale.region(), maximum)
    }
    append_recipe_region(output, launch.activation_scale_workspace(), maximum)
    match launch.kernel_workspace() {
      None => append_recipe_text(output, "workspace-absent", maximum)
      Some(region) => append_recipe_region(output, region, maximum)
    }
  }
  Bytes::from_array(output)
}

///|
/// Offline vision program export. Current text Resident does not bind this
/// module; this report never claims that image requests execute on a device.
async fn prepare_vision_sources(
  model : @model.Spec,
  bound : @weights.BoundWeights,
  output : String,
) -> Json {
  let program = @vision.Program::new(
    model,
    maximum_patches=16384,
    maximum_crop_patches=4096,
  )
  let names = program.weight_names()
  let mut weight_bytes = 0L
  for name in names {
    guard bound.find_tensor(name) is Some(binding) else {
      fail("missing vision tensor " + name)
    }
    weight_bytes += binding.byte_length
  }
  let entries : ReadOnlyArray[Json] = program
    .entries()
    .map(entry => {
      "symbol": Json(entry.symbol()),
      "launch": Json(entry.launch()),
      "parameters": Json(entry.parameters()),
      "static_shared_bytes": Json(entry.shared_bytes()),
    })
  let stages : ReadOnlyArray[Json] = program
    .stages()
    .map(stage => {
      "entry": Json(stage.entry()),
      "parameters": Json(stage.arguments().length()),
    })
  @fs.write_file(
    output + "/vision_encoder.cu",
    program.source_bytes(),
    create_mode=CreateNew,
  )
  {
    "source": "vision_encoder.cu",
    "scope": "prepared AOT vision program; text Resident integration pending",
    "row_capacities_raw_encoded_output": Json(program.row_capacities()),
    "maximum_crop_patches": 4096,
    "workspace_bytes": Json(program.workspace_bytes()),
    "workspace_scope": "eight BF16 scratch regions; pixels, features and integer metadata are external",
    "mutable_storage_bytes": Json(program.storage_bytes()),
    "native_frame_software_implemented": true,
    "text_resident_integrated": false,
    "weight_tensor_count": Json(names.length()),
    "weight_bytes": Json(weight_bytes),
    "weight_names": Json(names),
    "entrypoints": Json(entries),
    "stages": Json(stages),
    "runtime_bound": false,
    "cuda_compiled": false,
    "device_qualified": false,
  }
}

///|
/// Export the same immutable program consumed by the native frame binder.
/// No separate command-line ABI or workspace calculation is maintained.
async fn prepare_expert_sources(
  operators : @operators.DecoderOperators,
  program : @moe.Program,
  weights : @weights.BoundWeights,
  layer_count : Int32,
  output : String,
) -> Json {
  let plan = operators.routed_experts()
  @fs.write_file(
    output + "/moe.cu",
    program.source_bytes(),
    create_mode=CreateNew,
  )
  let stages : ReadOnlyArray[Json] = program
    .stages()
    .map(stage => {
      (
        {
          "symbol": Json(stage.symbol()),
          "launch": Json(stage.launch()),
          "arguments": Json(stage.arguments().map(slot => slot.name())),
        } : Json)
    })
  let regions : ReadOnlyArray[Json] = program
    .regions()
    .map(region => {
      (
        {
          "slot": Json(region.slot.name()),
          "workspace_offset": match region.workspace_offset {
            Some(offset) => Json(offset)
            None => Json::null()
          },
          "byte_count": Json(region.byte_count),
          "alignment": Json(region.alignment),
        } : Json)
    })
  let layers : Array[Json] = []
  let mut upload_count = 0
  for layer in 0..<layer_count {
    let copies = operators.expert_uploads(weights, layer~)
    upload_count += copies.length()
    layers.push({
      "layer": Json(layer),
      "bank_bytes": Json(plan.bank_bytes()),
      "copies": Json(
        copies.map(copy => {
          (
            {
              "shard_index": Json(copy.shard_index),
              "file_offset": Json(copy.file_offset),
              "bank_offset": Json(copy.bank_offset),
              "byte_length": Json(copy.byte_length),
            } : Json)
        }),
      ),
    })
  }
  @fs.write_file(
    output + "/expert_uploads.json",
    @utf8.encode(Json(layers).stringify()),
    create_mode=CreateNew,
  )
  {
    "source": "moe.cu",
    "stages": Json(stages),
    "regions": Json(regions),
    "workspace_bytes": Json(program.workspace_bytes()),
    "workspace_ownership": "caller-owned arena, borrowed by frames, reused serially across layers",
    "native_frame_implemented": true,
    "native_frame_executed": false,
    "geometry": Json(plan.geometry()),
    "expert_bank_bytes_per_layer": Json(plan.bank_bytes()),
    "expert_bank_bytes_all_layers": Json(
      plan.bank_bytes() * layer_count.to_int64(),
    ),
    "expert_stride_bytes": Json(plan.expert_stride()),
    "expert_weight_offsets": Json(plan.weight_offsets()),
    "route_weight_storage": "FP32 carrying exactly BF16-rounded values",
    "combine": "ascending expert ID, BF16 weighted output and BF16 after every contribution; F32 buffer carries exact BF16 values",
    "upload_file": "expert_uploads.json",
    "upload_count": Json(upload_count),
    "upload_strategy": "bounded source ranges copied directly into one destination bank; no persistent second weight copy or transpose",
  }
}

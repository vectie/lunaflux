///|
fn media_plan_receipt(
  model : @bootstrap.PreparedImageModel,
  requests : ReadOnlyArray[@media.PreparedRequest],
  budget : @image_runtime.Budget,
) -> Json {
  let summaries : ReadOnlyArray[Json] = requests.map(request => {
    "id": Json(request.id),
    "input_tokens": Json(request.input.prompt().tokens().length()),
    "token_ids": Json(request.input.prompt().tokens()),
    "image_count": Json(request.input.image_count()),
    "patch_rows": Json(request.input.patch_rows()),
    "feature_rows": Json(request.input.feature_rows()),
    "prepared_pixel_bytes": Json(request.input.pixel_bytes()),
    "maximum_new_tokens": Json(request.maximum_new_tokens),
    "context_ceiling": Json(request.context_ceiling),
  })
  {
    "format": "lunaflux.minicpm-v47-media-plan.v1",
    "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
    "revision": "98e79ba972b586ac6146d3505771ce1467e199ba",
    "architecture": "sm_121",
    "compiler_profile": "unqualified software profile",
    "context_tokens": 4096,
    "maximum_rows": 32,
    "sequence_slots": 1,
    "aot_module_count": 9,
    "sampling": "explicit greedy; not upstream stochastic default",
    "inspected_shards": Json(model.text.inspected_shards),
    "inspected_tensors": Json(model.text.inspected_tensors),
    "text_weight_bytes": Json(budget.text.weight_bytes),
    "vision_weight_bytes": Json(budget.vision_weight_bytes),
    "mutable_output_metadata_bytes": Json(budget.mutable_bytes),
    "total_allocation_bytes": Json(budget.total_allocation_bytes),
    "requests": Json(summaries),
    "weight_payload_read": false,
    "media_decoded_and_prepared": true,
    "gpu_execution": false,
    "device_qualified": false,
  }
}

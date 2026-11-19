///|
async fn write_receipt(output : String, name : String, document : Json) -> Unit {
  @fs.write_file(
    output + "/" + name,
    @utf8.encode(document.stringify(indent=2)),
    create_mode=CreateNew,
  )
}

///|
/// Token limits can stop between UTF-8 byte pieces. Preserve the token IDs in
/// the receipt and identify replacement decoding instead of losing the result.
fn decode_output(
  tokenizer : @tokenizer.TokenizerSpec,
  tokens : ArrayView[Int32],
) -> (String, Bool) raise {
  let bytes = tokenizer.decode_bytes(tokens, special_policy=Preserve)
  try @utf8.decode(bytes) catch {
    _ => (@utf8.decode_lossy(bytes), false)
  } noraise {
    text => (text, true)
  }
}

///|
fn plan_receipt(
  model : @bootstrap.PreparedModel,
  requests : ReadOnlyArray[@bootstrap.PreparedRequest],
) -> Json raise {
  let budget = @runtime.Budget::for_text(
    model.program,
    weight_bytes=model.weights.byte_count(),
  )
  {
    "format": "lunaflux.minicpm-v47-text-plan.v1",
    "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
    "revision": "98e79ba972b586ac6146d3505771ce1467e199ba",
    "identity_scope": "deployment-staged association; no payload hash",
    "architecture": "sm_121",
    "compiler_profile": "unqualified software profile",
    "maximum_rows": 32,
    "context_tokens": 4096,
    "sequence_slots": 1,
    "sampling": "explicit greedy; not the upstream stochastic default",
    "inspected_shards": Json(model.inspected_shards),
    "inspected_tensors": Json(model.inspected_tensors),
    "text_allocations": Json(model.weights.allocations().length()),
    "weight_bytes": Json(budget.weight_bytes),
    "storage_bytes": Json(budget.storage_bytes),
    "output_bytes": Json(budget.output_bytes),
    "total_allocation_bytes": Json(budget.total_allocation_bytes),
    "requests": Json(
      requests.map(request => {
        "id": Json(request.id),
        "input_tokens": Json(request.prompt.tokens().length()),
        "token_ids": Json(request.prompt.tokens()),
        "maximum_new_tokens": Json(request.maximum_new_tokens),
        "context_ceiling": Json(request.context_ceiling),
      }),
    ),
    "weight_payload_read": false,
    "gpu_execution": false,
    "device_qualified": false,
  }
}

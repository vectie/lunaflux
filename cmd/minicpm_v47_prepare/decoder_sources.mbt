///|
async fn prepare_decoder_sources(
  program : @decoder.Program,
  output : String,
) -> Json {
  @fs.write_file(
    output + "/decoder_envelope.cu",
    program.source_bytes(),
    create_mode=CreateNew,
  )
  let budget = program.budget()
  {
    "envelope_source": "decoder_envelope.cu",
    "layer_count": Json(program.layers().length()),
    "operations_per_layer": Json(
      program.operations().map(operation => operation.name()),
    ),
    "prefill_launches": Json(program.launch_count(Prefill)),
    "decode_launches": Json(program.launch_count(Decode)),
    "layers": Json(
      program
      .layers()
      .map(layer => {
        "index": Json(layer.index),
        "kind": Json(
          match layer.kind {
            GatedDeltaLinearAttention => "recurrent"
            FullAttention => "full_attention"
          },
        ),
        "state_offset": Json(layer.state_offset),
        "convolution_offset": match layer.convolution_offset {
          Some(offset) => Json(offset)
          None => Json::null()
        },
      }),
    ),
    "buffer_budget": {
      "bytes_per_activation_bank": Json(budget.activation_bytes),
      "activation_banks": 3,
      "branch_workspace_bytes": Json(budget.branch_workspace_bytes),
      "moe_workspace_bytes": Json(budget.moe_workspace_bytes),
      "convolution_state_bytes": Json(budget.convolution_state_bytes),
      "recurrent_state_bytes": Json(budget.recurrent_state_bytes),
      "key_cache_bytes": Json(budget.key_cache_bytes),
      "value_cache_bytes": Json(budget.value_cache_bytes),
    },
    "residual_contract": "centered input norm; branch; BF16 residual; centered post-attention norm; MoE; BF16 residual",
    "workspace_contract": "one shared attention/recurrent arena and one separate shared MoE arena; retain the startup expert map across branch execution",
    "native_frame_implemented": true,
    "cuda_device_frame_executed": false,
    "includes_embeddings_and_logits": false,
    "cuda_compiled": false,
    "device_qualified": false,
  }
}

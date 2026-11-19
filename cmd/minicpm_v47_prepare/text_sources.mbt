///|
async fn prepare_text_sources(program : @text.Program, output : String) -> Json {
  @fs.write_file(
    output + "/text_io.cu",
    program.source_bytes(),
    create_mode=CreateNew,
  )
  let source = program.text_source()
  let plan = source.plan()
  let symbols = []
  for stage in 0..<4 {
    symbols.push(source.symbol(stage))
  }
  {
    "source": "text_io.cu",
    "symbols": Json(symbols),
    "geometry": Json(plan.geometry()),
    "prefill_launches_including_decoder": Json(program.launch_count(Prefill)),
    "decode_launches_including_decoder": Json(program.launch_count(Decode)),
    "embedding_table_bytes": Json(plan.table_bytes()),
    "language_head_bytes": Json(source.head_bytes()),
    "tables_tied": false,
    "normalized_output_rows_bytes": Json(plan.normalized_bytes()),
    "logits_bytes": Json(plan.logits_bytes()),
    "workspace_bytes": Json(program.workspace_bytes()),
    "mutable_owned_device_bytes": Json(program.storage_budget().owned_bytes),
    "private_output_workspace_bytes": Json(
      program.storage_budget().output_workspace_bytes,
    ),
    "storage_owner": "integration/minicpm_v47_text.Storage owns state/activation/metadata/token allocations; Frame owns private text output workspace",
    "output_rows": "caller supplies output_count and selected_rows, independently of decoder input rows",
    "final_norm": "FP32 normalization and one-plus-gamma, then BF16 publication",
    "logits": "existing shape-aware head; BF16-rounded values widened exactly into F32 storage",
    "selection": "existing greedy reduction; full selected-row logits remain available",
    "native_frame_implemented": true,
    "cuda_device_frame_executed": false,
    "cuda_compiled": false,
    "device_qualified": false,
  }
}

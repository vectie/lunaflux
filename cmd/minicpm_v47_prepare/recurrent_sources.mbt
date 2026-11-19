///|
async fn prepare_recurrent_sources(
  program : @recurrent.Program,
  output : String,
) -> Json {
  @fs.write_file(
    output + "/recurrent_branch.cu",
    program.source_bytes(),
    create_mode=CreateNew,
  )
  {
    "source": "recurrent_branch.cu",
    "ordered_stages": Json(
      program
      .stages()
      .map(stage => {
        "symbol": Json(stage.symbol()),
        "launch": Json(stage.launch()),
        "arguments": Json(stage.arguments().map(slot => slot.name())),
      }),
    ),
    "regions": Json(
      program
      .regions()
      .map(region => {
        "slot": Json(region.slot.name()),
        "workspace_offset": match region.workspace_offset {
          Some(offset) => Json(offset)
          None => Json::null()
        },
        "bytes": Json(region.byte_count),
        "alignment": Json(region.alignment),
      }),
    ),
    "workspace_bytes": Json(program.workspace_bytes()),
    "workspace_ownership": "one borrowed scratch arena reused serially across recurrent layers; persistent BF16 history and FP32 matrices remain outside",
    "convolution_arithmetic": "BF16 convolution result before SiLU, matching upstream unfused fallback; fused-kernel alternatives require comparison",
    "beta": "BF16 sigmoid",
    "log_decay": "FP32 negative exp(A_log) times threshold-20 softplus(a+dt_bias); no clipping",
    "output_norm": "direct gamma; BF16 norm and weight rounds; SiLU gate",
    "state_contract": "caller supplies exclusive request slots, valid CSR and identical reset flags for convolution and recurrence",
    "native_frame_implemented": true,
    "native_frame_executed": false,
    "cuda_compiled": false,
    "device_qualified": false,
  }
}

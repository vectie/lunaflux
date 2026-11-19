///|
async fn prepare_multimodal_sources(
  text : @text.Program,
  output : String,
) -> Json {
  let program = @multimodal.Program::new(text)
  let source = program.scatter_source()
  @fs.write_file(
    output + "/image_scatter.cu",
    source.source_bytes(),
    create_mode=CreateNew,
  )
  {
    "source": "image_scatter.cu",
    "symbol": Json(source.symbol()),
    "launch": Json(source.launch()),
    "parameters": Json(source.parameter_count()),
    "prefill_launches": Json(program.launch_count(Prefill)),
    "decode_launches": Json(program.launch_count(Decode)),
    "metadata_bytes": Json(program.metadata_bytes()),
    "scope": "native prefill feature insertion, combined residency, local and buffered HTTP media; device qualification pending",
    "feature_transfer": "device BF16 word copy; no host feature readback",
    "native_frame_implemented": true,
    "combined_resident_integrated": true,
    "cuda_compiled": false,
    "device_qualified": false,
  }
}

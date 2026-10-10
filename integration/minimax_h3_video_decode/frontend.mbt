///|
/// One already-selected temporal/spatial tile, not the complete tiled decoder.
pub struct Frontend {
  source : Bytes
  symbols : ReadOnlyArray[String]
  input_bytes : Int64
  output_bytes : Int64
  rows_per_batch : Int
  weights : @minimax_h3_weights.VideoFrontendWeightPacking
}

///|
/// Accept destandardized contiguous [B,24,T,H,W] F32. Output is
/// [B,T*H*W+5,2048] F32, including learned registers and the zero CLS row.
pub fn Frontend::new(
  batch~ : Int,
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  prefix~ : String,
  max_workspace_bytes~ : Int64,
) -> Self raise {
  guard temporal > 0 &&
    height > 0 &&
    width > 0 &&
    temporal.to_int64() * height.to_int64() <= 2147483647L / width.to_int64() else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let positions = temporal * height * width
  let symbols = [prefix + "_post_quant", prefix + "_embedding"]
  let embedding = @video_latent_embedding_cuda_source.render(
    batch~,
    positions~,
    input_channels=24,
    hidden_channels=2048,
    register_rows=4,
    symbol=symbols[1],
  )
  let input_bytes = embedding.input_elements * 4L
  let output_bytes = embedding.output_elements * 4L
  guard input_bytes <= max_workspace_bytes &&
    output_bytes <= max_workspace_bytes - input_bytes else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let post_quant = @channel_major_projection_cuda_source.f32_pointwise_conv1d_source(
    batch_count=batch,
    input_channels=24,
    output_channels=24,
    time_count=positions,
    function_symbol=symbols[0],
  )
  {
    source: @cuda_translation_unit.compose([
      post_quant.source_bytes(),
      embedding.source,
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    input_bytes,
    output_bytes,
    rows_per_batch: positions + 5,
    weights: @minimax_h3_weights.video_frontend_weight_packing(
      max_bytes=239968L,
    ),
  }
}

///|
/// Contiguous F32 [1,3,T,H,W], inverse ImageNet normalization and top-left crop.
pub struct VideoOutput {
  source : Bytes
  symbol : String
  frames : Int
  height : Int
  width : Int
  input_bytes : Int64
  output_bytes : Int64
}

///|
pub fn VideoOutput::new(
  frames~ : Int,
  input_height~ : Int,
  input_width~ : Int,
  height~ : Int,
  width~ : Int,
  symbol~ : String,
) -> Self raise {
  guard frames > 0 &&
    height > 0 &&
    width > 0 &&
    input_height >= height &&
    input_width >= width &&
    frames.to_int64() * input_height.to_int64() <=
    2147483647L / 3L / input_width.to_int64() else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let _ = @video_latent_embedding_cuda_source.tile_copy_source(
    elements=1L,
    symbol~,
  )
  let input_bytes = frames.to_int64() *
    input_height.to_int64() *
    input_width.to_int64() *
    12L
  let output_bytes = frames.to_int64() *
    height.to_int64() *
    width.to_int64() *
    12L
  // torchvision converts the inverse transform's Double constants to F32,
  // then subtracts and divides. Do not replace with x*std+mean or FMA.
  let means = [0.485, 0.456, 0.406]
  let stds = [0.229, 0.224, 0.225]
  let inv_means = Array::makei(3, i => {
    (-means[i] / stds[i]).to_float().to_double().to_string() + "f"
  })
  let inv_stds = Array::makei(3, i => {
    (1.0 / stds[i]).to_float().to_double().to_string() + "f"
  })
  let source = @utf8.encode(
    (
      $|extern "C" __global__ void \{symbol}(const float* input,float* output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{output_bytes/4L}ULL)return;
      $| const unsigned long long x=i%\{width}ULL,y=(i/\{width}ULL)%\{height}ULL,t=(i/(\{width}ULL*\{height}ULL))%\{frames}ULL,c=i/(\{width}ULL*\{height}ULL*\{frames}ULL);
      $| const float m[3]={\{inv_means[0]},\{inv_means[1]},\{inv_means[2]}},s[3]={\{inv_stds[0]},\{inv_stds[1]},\{inv_stds[2]}};
      $| float v=__fdiv_rn(__fsub_rn(input[((c*\{frames}ULL+t)*\{input_height}ULL+y)*\{input_width}ULL+x],m[c]),s[c]);
      $| output[i]=v<0.0f?0.0f:(v>1.0f?1.0f:v);
      $|}
    ),
  )
  { source, symbol, frames, height, width, input_bytes, output_bytes, }
}

///|
pub struct VideoDecodeBundle {
  source : Bytes
  input : @minimax_h3_block_source.VideoInputSource
  decoder : TiledDecoder
  output : VideoOutput
  workspace_bytes : Int64
}

///|
pub fn VideoDecodeBundle::new(
  input : @minimax_h3_block_source.VideoInputSource,
  decoder : TiledDecoder,
  max_workspace_bytes~ : Int64,
) -> Self raise {
  let request = input.plan.request()
  guard decoder.input_bytes == input.value_bytes &&
    decoder.temporal.latent_frames == request.video_latent_frames() &&
    decoder.output_height / 16 == request.video_latent_height() &&
    decoder.output_width / 16 == request.video_latent_width() &&
    decoder.temporal.output_frames == request.frames() else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let output = VideoOutput::new(
    frames=request.frames(),
    input_height=decoder.output_height,
    input_width=decoder.output_width,
    height=request.height(),
    width=request.width(),
    symbol="lunaflux_video_canonical_output_v1",
  )
  guard decoder.workspace_bytes <= max_workspace_bytes &&
    input.value_bytes <= (max_workspace_bytes - decoder.workspace_bytes) / 2L &&
    output.output_bytes <=
    max_workspace_bytes - decoder.workspace_bytes - input.value_bytes * 2L else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  {
    source: @cuda_translation_unit.compose([
      input.source,
      decoder.source,
      output.source,
    ]),
    input,
    decoder,
    output,
    workspace_bytes: decoder.workspace_bytes +
    input.value_bytes * 2L +
    output.output_bytes,
  }
}

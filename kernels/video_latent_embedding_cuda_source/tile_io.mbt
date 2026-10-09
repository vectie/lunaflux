///|
/// Gather a 7-frame latent tile; temporal padding repeats the final latent.
pub fn tile_gather_source(
  latent_t~ : Int,
  latent_h~ : Int,
  latent_w~ : Int,
  start_t~ : Int,
  start_y~ : Int,
  start_x~ : Int,
  tile_h~ : Int,
  tile_w~ : Int,
  symbol~ : String,
) -> Bytes raise EmbeddingError {
  guard latent_t > 0 &&
    latent_h > 0 &&
    latent_w > 0 &&
    start_t >= 0 &&
    start_y >= 0 &&
    start_x >= 0 &&
    tile_h > 0 &&
    tile_w > 0 &&
    tile_h <= latent_h - start_y &&
    tile_w <= latent_w - start_x &&
    tile_h.to_int64() * tile_w.to_int64() <= 2147483647L / 168L &&
    latent_t.to_int64() * latent_h.to_int64() <=
    9223372036854775807L / 24L / latent_w.to_int64() else {
    raise InvalidGeometry
  }
  let checked = render(
    batch=1,
    positions=tile_h * tile_w * 168,
    input_channels=1,
    hidden_channels=1,
    register_rows=0,
    symbol~,
  )
  let _ = checked
  @utf8.encode(
    (
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const uint32_t *input,uint32_t *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{tile_h.to_int64()*tile_w.to_int64()*168L}ULL)return;
      $| const unsigned long long x=i%\{tile_w}ULL,y=(i/\{tile_w}ULL)%\{tile_h}ULL,t=(i/(\{tile_w}ULL*\{tile_h}ULL))%7ULL,c=i/(\{tile_w}ULL*\{tile_h}ULL*7ULL);
      $| const unsigned long long st=t+\{start_t}ULL<\{latent_t}ULL?t+\{start_t}ULL:\{latent_t-1}ULL;
      $| output[i]=input[((c*\{latent_t}ULL+st)*\{latent_h}ULL+y+\{start_y}ULL)*\{latent_w}ULL+x+\{start_x}ULL];
      $|}
    ),
  )
}

///|
pub fn tile_copy_source(
  elements~ : Int64,
  symbol~ : String,
) -> Bytes raise EmbeddingError {
  guard elements > 0L && elements < 2147483647L else { raise InvalidGeometry }
  let checked = render(
    batch=1,
    positions=elements.to_int(),
    input_channels=1,
    hidden_channels=1,
    register_rows=0,
    symbol~,
  )
  let _ = checked
  @utf8.encode(
    (
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const uint32_t *input,uint32_t *output){const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i<\{elements}ULL)output[i]=input[i];}
    ),
  )
}

///|
pub fn temporal_publish_source(
  height~ : Int,
  width~ : Int,
  chunk~ : Int,
  chunks~ : Int,
  available_frames~ : Int,
  output_frames~ : Int,
  symbol~ : String,
) -> Bytes raise EmbeddingError {
  guard height > 0 &&
    width > 0 &&
    chunks > 0 &&
    chunk >= 0 &&
    chunk < chunks &&
    output_frames > 0 &&
    output_frames <= available_frames &&
    height.to_int64() * width.to_int64() <= 2147483647L / 84L &&
    output_frames.to_int64() * height.to_int64() <=
    9223372036854775807L / 3L / width.to_int64() else {
    raise InvalidGeometry
  }
  let count = if chunk == chunks - 1 { 22 } else { 17 }
  let checked = render(
    batch=1,
    positions=count * 3 * height * width,
    input_channels=1,
    hidden_channels=1,
    register_rows=0,
    symbol~,
  )
  let _ = checked
  @utf8.encode(
    (
      $|#include <math.h>
      $|extern "C" __global__ void \{symbol}(const float *current,const float *previous,float *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{count.to_int64()*3L*height.to_int64()*width.to_int64()}ULL)return;
      $| const unsigned long long spatial=i%(\{height}ULL*\{width}ULL),part=(i/(\{height}ULL*\{width}ULL))%\{count}ULL,c=i/(\{height}ULL*\{width}ULL*\{count}ULL);
      $| const unsigned long long logical=\{chunk.to_int64()*17L}ULL+part;
      $| if(logical<\{available_frames-output_frames}ULL||logical>=\{available_frames}ULL)return;
      $| const unsigned long long frame=part<17ULL?part+3ULL:part-17ULL+23ULL;
      $| float value=current[(c*28ULL+frame)*\{height}ULL*\{width}ULL+spatial];
      $| if(\{chunk}>0&&part<5ULL){const float b=__fdiv_rn((float)part,5.0f),a=__fsub_rn(1.0f,b);value=__fadd_rn(__fmul_rn(previous[(c*28ULL+23ULL+part)*\{height}ULL*\{width}ULL+spatial],a),__fmul_rn(value,b));}
      $| output[(c*\{output_frames}ULL+logical-\{available_frames-output_frames}ULL)*\{height}ULL*\{width}ULL+spatial]=value;
      $|}
    ),
  )
}

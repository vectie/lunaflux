///|
pub fn spatial_assemble_source(
  height~ : Int,
  width~ : Int,
  tile_h~ : Int,
  tile_w~ : Int,
  y_starts : ArrayView[Int],
  x_starts : ArrayView[Int],
  symbol~ : String,
) -> Bytes raise EmbeddingError {
  guard height > 0 &&
    width > 0 &&
    tile_h > 0 &&
    tile_w > 0 &&
    !y_starts.is_empty() &&
    !x_starts.is_empty() &&
    height.to_int64() * width.to_int64() <= 2147483647L / 84L else {
    raise InvalidGeometry
  }
  for
    (starts, extent, tile) in [
      (y_starts, height, tile_h),
      (x_starts, width, tile_w),
    ] {
    guard starts[0] == 0 &&
      starts[starts.length() - 1].to_int64() + tile.to_int64() ==
      extent.to_int64() else {
      raise InvalidGeometry
    }
    for i in 1..<starts.length() {
      guard starts[i] > starts[i - 1] && starts[i] - starts[i - 1] <= tile else {
        raise InvalidGeometry
      }
    }
  }
  let checked = render(
    batch=1,
    positions=height * width * 84,
    input_channels=1,
    hidden_channels=1,
    register_rows=0,
    symbol~,
  )
  let _ = checked
  let ys = y_starts.map(v => v.to_string()).join(",")
  let xs = x_starts.map(v => v.to_string()).join(",")
  @utf8.encode(
    (
      $|#include <math.h>
      $|extern "C" __global__ void \{symbol}(const float *tiles,float *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{height.to_int64()*width.to_int64()*84L}ULL)return;
      $| const int ys[\{y_starts.length()}]={\{ys}},xs[\{x_starts.length()}]={\{xs}};
      $| const unsigned long long x=i%\{width}ULL,y=(i/\{width}ULL)%\{height}ULL,plane=i/(\{width}ULL*\{height}ULL);
      $| int iy=0,ix=0;for(int j=1;j<\{y_starts.length()};++j)if(y>=(unsigned)ys[j])iy=j;for(int j=1;j<\{x_starts.length()};++j)if(x>=(unsigned)xs[j])ix=j;
      $| const unsigned long long ly=y-ys[iy],lx=x-xs[ix],tile=(unsigned long long)iy*\{x_starts.length()}ULL+ix;
      $| float value=tiles[((tile*84ULL+plane)*\{tile_h}ULL+ly)*\{tile_w}ULL+lx];
      $| if(iy>0){const int overlap=ys[iy-1]+\{tile_h}-ys[iy];if(ly<(unsigned)overlap){const float b=__fdiv_rn((float)ly,(float)overlap),a=__fsub_rn(1.0f,b);const float previous=tiles[(((tile-\{x_starts.length()}ULL)*84ULL+plane)*\{tile_h}ULL+\{tile_h}ULL-overlap+ly)*\{tile_w}ULL+lx];value=__fadd_rn(__fmul_rn(previous,a),__fmul_rn(value,b));}}
      $| if(ix>0){const int overlap=xs[ix-1]+\{tile_w}-xs[ix];if(lx<(unsigned)overlap){const float b=__fdiv_rn((float)lx,(float)overlap),a=__fsub_rn(1.0f,b);const float previous=tiles[(((tile-1ULL)*84ULL+plane)*\{tile_h}ULL+ly)*\{tile_w}ULL+\{tile_w}ULL-overlap+lx];value=__fadd_rn(__fmul_rn(previous,a),__fmul_rn(value,b));}}
      $| output[i]=value;
      $|}
    ),
  )
}

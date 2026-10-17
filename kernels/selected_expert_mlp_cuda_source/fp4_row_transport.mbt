///|
/// Materialize a quantized row view without decoding or requantizing. The
/// original activation/route order remains the publication order. Invalid
/// padded rows receive zero bytes in both planes; global scale is borrowed.
fn fp4_pack_rows_source(
  packed : @matrix.PackedExpertRows,
  product : Bool,
  jobs : Int,
  selected : Int,
  metadata : Int64,
  source_scales : Int64,
  symbol : String,
) -> String {
  let (payload, scales, scale_offset) = packed.plane(product)
  let first_source = if product { "job" } else { "job/" + selected.to_string() }
  (
    $|extern "C" __global__ void \{symbol}(const int *counts,const unsigned char *input,__nv_bfloat16 *activation) {
    $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);
    $|  unsigned char *packed=(unsigned char *)activation+\{packed.base()}ULL;
    $|  for(int row=blockIdx.x;row<meta[0]*16;row+=gridDim.x) {
    $|    int job=meta[1+\{jobs}+row],source_row=job>=0?\{first_source}:0;
    $|    for(int col=threadIdx.x*16;col<\{payload};col+=blockDim.x*16) {
    $|      uint4 value=make_uint4(0,0,0,0);
    $|      if(job>=0) value=*(const uint4 *)(input+(unsigned long long)source_row*\{payload}+col);
    $|      *(uint4 *)(packed+(unsigned long long)row*\{payload}+col)=value;
    $|    }
    $|    for(int col=threadIdx.x;col<\{scales};col+=blockDim.x)
    $|      packed[\{scale_offset}ULL+(unsigned long long)row*\{scales}+col]=job>=0?input[\{source_scales}ULL+(unsigned long long)source_row*\{scales}+col]:0;
    $|  }
    $|}
  )
}

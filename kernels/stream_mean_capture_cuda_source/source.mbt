///|
pub struct StreamMeanCaptureCudaSource {
  priv plan : @precision.StreamMeanCapturePrecision
  priv symbol : String
  priv bytes : Bytes
  priv grid : Int
}

///|
pub fn stream_mean_capture_source(
  plan : @precision.StreamMeanCapturePrecision,
  symbol~ : String,
) -> StreamMeanCaptureCudaSource raise @precision.PrecisionPlanError {
  guard !symbol.is_empty() &&
    symbol.length() <= 100 &&
    symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise IllegalPlacement
  }
  let (streams, hidden, rows, segments) = plan.geometry()
  let elements = rows.to_int64() * hidden.to_int64()
  let grid = elements / 256L + (if elements % 256L == 0L { 0L } else { 1L })
  guard grid <= 2147483647L else { raise MemoryBudgetExceeded }
  let source = Array::makei(segments, segment => {
    render_segment(
      symbol + "_segment_" + segment.to_string(),
      streams,
      hidden,
      rows,
      segments,
      segment,
    )
  }).join("\n")
  { plan, symbol, bytes: @utf8.encode(source), grid: grid.to_int(), }
}

///|
fn render_segment(
  symbol : String,
  streams : Int,
  hidden : Int,
  rows : Int,
  segments : Int,
  segment : Int,
) -> String {
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbol}(const int32_t* counts, const __nv_bfloat16* input, __nv_bfloat16* output) {
    $|  const int live = counts[3];
    $|  if(live <= 0 || live > \{rows}) return;
    $|  const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;
    $|  if(i >= (unsigned long long)live*\{hidden}ULL) return;
    $|  const unsigned long long row=i/\{hidden}ULL, column=i%\{hidden}ULL;
    $|  float sum=0.0f;
    $|  for(int s=0;s<\{streams};++s) sum=__fadd_rn(sum,__bfloat162float(input[(row*\{streams}ULL+s)*\{hidden}ULL+column]));
    $|  output[(row*\{segments}ULL+\{segment}ULL)*\{hidden}ULL+column]=__float2bfloat16_rn(__fdiv_rn(sum,\{streams}.0f));
    $|}
    $|
  source
}

///|
pub fn StreamMeanCaptureCudaSource::plan(
  self : Self,
) -> @precision.StreamMeanCapturePrecision {
  self.plan
}

///|
pub fn StreamMeanCaptureCudaSource::source_bytes(self : Self) -> Bytes {
  self.bytes
}

///|
pub fn StreamMeanCaptureCudaSource::symbol(
  self : Self,
  segment : Int,
) -> String raise @precision.PrecisionPlanError {
  let (_, _, _, count) = self.plan.geometry()
  guard segment >= 0 && segment < count else { raise IllegalPlacement }
  self.symbol + "_segment_" + segment.to_string()
}

///|
pub fn StreamMeanCaptureCudaSource::grid_x(self : Self) -> Int {
  self.grid
}

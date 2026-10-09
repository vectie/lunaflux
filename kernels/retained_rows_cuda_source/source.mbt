///|
pub struct RetainedRowsCudaSource {
  priv plan : @precision.RetainedRowsPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn retained_rows_source(
  plan : @precision.RetainedRowsPrecision,
  symbol_prefix~ : String,
) -> RetainedRowsCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (width, stride, rows, capacity) = plan.geometry()
  let symbols = [
    symbol_prefix + "_reserve",
    symbol_prefix + "_copy",
    symbol_prefix + "_publish",
  ]
  let source = @utf8.encode(
    (
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const int32_t *reset, const int32_t *upstream_append, int32_t *history, int32_t *cache_counts, int32_t *append) {
      $|  if (threadIdx.x || blockIdx.x) return;
      $|  for (int i = 0; i < 5; ++i) cache_counts[i] = 0;
      $|  if (reset[0] == 1) { history[0] = 0; history[1] = 0; }
      $|  const int base = history[0], live = counts[3]; append[0] = base; append[1] = 0; append[2] = 1;
      $|  if (reset[0] < 0 || reset[0] > 1 || upstream_append[2] || history[1] || base < 0 || base > \{capacity} || live < 0 || live > \{rows} || live > \{capacity} - base || (live > 0 && counts[2] != 1)) { history[1] = 1; return; }
      $|  for (int row = 0; row < live; ++row) if (positions[row] != (base + row) * \{stride}) { history[1] = 1; return; }
      $|  append[1] = live; append[2] = 0;
      $|}
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *append, const uint16_t *input, uint16_t *cache) {
      $|  const int row = (int)blockIdx.x; if (append[2] || row >= append[1]) return;
      $|  const unsigned long long source = (unsigned long long)row * \{width}ULL, destination = (unsigned long long)(append[0] + row) * \{width}ULL;
      $|  for (int col = threadIdx.x; col < \{width}; col += blockDim.x) cache[destination + col] = input[source + col];
      $|}
      $|extern "C" __global__ void \{symbols[2]}(const int32_t *append, int32_t *history, int32_t *cache_counts) {
      $|  if (threadIdx.x || blockIdx.x || append[2]) return;
      $|  const int length = append[0] + append[1]; history[0] = length;
      $|  cache_counts[2] = length ? 1 : 0; cache_counts[3] = length;
      $|}
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn RetainedRowsCudaSource::plan(
  self : Self,
) -> @precision.RetainedRowsPrecision {
  self.plan
}

///|
pub fn RetainedRowsCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RetainedRowsCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RetainedRowsCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (_, _, rows, _) = self.plan.geometry()
  if stage == 1 {
    (rows, 1, 256)
  } else {
    (1, 1, 1)
  }
}

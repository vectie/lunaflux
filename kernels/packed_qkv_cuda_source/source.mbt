///|
pub fn packed_qkv_source(
  plan : @precision.PackedQkvPrecision,
  symbol~ : String,
) -> Bytes raise {
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (query, kv, rows) = plan.geometry()
  let width = query + kv * 2
  let text =
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const uint16_t *query, const uint16_t *key, const uint16_t *value, uint16_t *output) {
    $|  const int row=blockIdx.x;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]) return;
    $|  for(unsigned long long column=threadIdx.x;column<\{width}ULL;column+=blockDim.x) {
    $|    const uint16_t word=column<\{query}ULL?query[(unsigned long long)row*\{query}ULL+column]:
    $|      column<\{query+kv}ULL?key[(unsigned long long)row*\{kv}ULL+column-\{query}ULL]:value[(unsigned long long)row*\{kv}ULL+column-\{query+kv}ULL];
    $|    output[(unsigned long long)row*\{width}ULL+column]=word;
    $|  }
    $|}
  @utf8.encode(text)
}

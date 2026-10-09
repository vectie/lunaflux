///|
/// Three ordered device effects: reserve/reset, parallel BF16 append, publish.
/// The publication occurs after all copy CTAs, never through a grid-wide race.
pub struct IndexedSparseHistoryCudaSource {
  priv plan : @precision.IndexedSparseHistoryPrecision
  priv symbols : ReadOnlyArray[String]
  priv source : Bytes
  priv copy_blocks : Int
}

///|
pub fn indexed_sparse_history_source(
  plan : @precision.IndexedSparseHistoryPrecision,
  symbol_prefix~ : String,
) -> IndexedSparseHistoryCudaSource raise {
  // Reuse the component symbol law; no independent string/identity policy.
  ignore(indexed_sparse_attention_source(plan.attention(), symbol_prefix~))
  let (rows, history) = plan.attention().capacity()
  let (_, dimension, _, _, _) = plan.attention().index_geometry()
  let (_, heads, keys, values) = plan.attention().attention_geometry()
  let kw = heads.to_int64() * keys.to_int64()
  let vw = heads.to_int64() * values.to_int64()
  let width = dimension.to_int64() * 2L + kw + vw
  let blocks = (rows.to_int64() * width + 255L) / 256L
  guard blocks <= 2147483647L else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  let symbols = [
    symbol_prefix + "_begin",
    symbol_prefix + "_copy",
    symbol_prefix + "_publish",
  ]
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *reset, int32_t *history, int32_t *valid, int32_t *append, int32_t *positions) {
    $|  if (blockIdx.x || threadIdx.x) return;
    $|  for (int i=0; i<\{rows}; ++i) positions[i]=-1;
    $|  int base=history[3], live=counts[3];
    $|  if (reset[0] || base==0) { base=0; for(int i=0; i<\{history}; ++i) valid[i]=0; }
    $|  history[0]=0; history[1]=0; history[2]=1; history[3]=base; history[4]=0;
    $|  append[0]=base; append[1]=0; append[2]=0;
    $|  if(base<0 || base>\{history} || live<0 || live>\{rows} || (live>0 && counts[2]!=1) || live>\{history}-base) { append[2]=1; history[3]=0; return; }
    $|  append[1]=live;
    $|  for(int row=0; row<live; ++row) positions[row]=base+row;
    $|}
    $|extern "C" __global__ void \{symbols[1]}(const int32_t *append, const __nv_bfloat16 *index_keys, const __nv_bfloat16 *gates, const __nv_bfloat16 *keys, const __nv_bfloat16 *values, __nv_bfloat16 *history_index, __nv_bfloat16 *history_gates, __nv_bfloat16 *history_keys, __nv_bfloat16 *history_values) {
    $|  if(append[2]) return;
    $|  const uint64_t cell=(uint64_t)blockIdx.x*256ULL+threadIdx.x, live=(uint64_t)append[1], base=(uint64_t)append[0];
    $|  const uint64_t iw=\{dimension}ULL, kw=\{kw}ULL, vw=\{vw}ULL;
    $|  if(cell<live*iw) history_index[base*iw+cell]=index_keys[cell];
    $|  else if(cell<live*iw*2ULL) { uint64_t i=cell-live*iw; history_gates[base*iw+i]=gates[i]; }
    $|  else if(cell<live*(iw*2ULL+kw)) { uint64_t i=cell-live*iw*2ULL; history_keys[base*kw+i]=keys[i]; }
    $|  else if(cell<live*(iw*2ULL+kw+vw)) { uint64_t i=cell-live*(iw*2ULL+kw); history_values[base*vw+i]=values[i]; }
    $|}
    $|extern "C" __global__ void \{symbols[2]}(const int32_t *append, int32_t *history, int32_t *valid) {
    $|  if(blockIdx.x || threadIdx.x || append[2]) return;
    $|  const int base=append[0], live=append[1];
    $|  for(int i=0; i<live; ++i) valid[base+i]=1;
    $|  history[3]=base+live;
    $|}
  {
    plan,
    symbols: ReadOnlyArray::from_array(symbols),
    source: @utf8.encode(source),
    copy_blocks: blocks.to_int(),
  }
}

///|
pub fn IndexedSparseHistoryCudaSource::plan(
  self : Self,
) -> @precision.IndexedSparseHistoryPrecision {
  self.plan
}

///|
pub fn IndexedSparseHistoryCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn IndexedSparseHistoryCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn IndexedSparseHistoryCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int) {
  if stage == 1 {
    (self.copy_blocks, 256)
  } else {
    (1, 1)
  }
}

///|
pub struct CommittedDraftKvCudaSource {
  priv plan : @precision.CommittedDraftKvPrecision
  priv phase : @precision.CommittedDraftPhase
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn committed_draft_kv_source(
  plan : @precision.CommittedDraftKvPrecision,
  phase~ : @precision.CommittedDraftPhase,
  symbol_prefix~ : String,
) -> CommittedDraftKvCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, window, main_rows, draft_rows, positions) = plan.geometry()
  guard window + draft_rows <= 4096 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let priming = if phase is Prime { 1 } else { 0 }
  let symbols = [
    symbol_prefix + "_reserve",
    symbol_prefix + "_copy",
    symbol_prefix + "_publish",
    symbol_prefix + "_attention",
  ]
  let reserve = if phase is PredictCommitted {
    (
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *main_counts, const int32_t *draft_counts, const int32_t *reset, const int32_t *start, const int32_t *history, int32_t *descriptor) {
      $|  if (threadIdx.x || blockIdx.x) return;
      $|  const int base = start[0], main = main_counts[3], draft = draft_counts[3];
      $|  descriptor[0] = 0; descriptor[1] = 0; descriptor[2] = 0; descriptor[3] = 1;
      $|  if (reset[0] < 0 || reset[0] > 1 || history[1] || base < 0 || base >= \{positions} || main < 1 || main > \{main_rows} || main > \{positions} - base || main_counts[2] != 1 || history[0] != base + main || draft < 1 || draft > \{draft_rows} || draft_counts[2] != 1 || draft > \{positions} - base - main) return;
      $|  descriptor[0] = base + main - 1; descriptor[2] = draft; descriptor[3] = 0;
      $|}
    )
  } else {
    (
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *main_counts, const int32_t *draft_counts, const int32_t *reset, const int32_t *start, int32_t *history, int32_t *descriptor) {
      $|  if (threadIdx.x || blockIdx.x) return;
      $|  if (reset[0] == 1) { history[0] = 0; history[1] = 0; }
      $|  const int base = start[0], main = main_counts[3], draft = \{priming} ? 0 : draft_counts[3];
      $|  descriptor[0] = base; descriptor[1] = 0; descriptor[2] = 0; descriptor[3] = 1;
      $|  if (reset[0] < 0 || reset[0] > 1 || history[1] || base < 0 || base >= \{positions} || base != history[0] || main < 1 || main > \{main_rows} || main > \{positions} - base || main_counts[2] != 1) { history[1] = 1; return; }
      $|  if (!\{priming} && (base == 0 || main != 1 || draft < 1 || draft > \{draft_rows} || draft_counts[2] != 1 || draft > \{positions} - base - main)) { history[1] = 1; return; }
      $|  descriptor[1] = main; descriptor[2] = draft; descriptor[3] = 0;
      $|}
    )
  }
  let publication = if phase is PredictCommitted {
    ""
  } else {
    (
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *descriptor, const uint16_t *main_kv, uint16_t *ring) {
      $|  const int slot = blockIdx.x; if (descriptor[3] || slot >= \{window}) return;
      $|  const int base = descriptor[0], end = base + descriptor[1];
      $|  const int latest = end - 1 - ((end - 1 - slot + \{window}) % \{window});
      $|  if (latest < base || latest >= end) return;
      $|  for (int component = threadIdx.x; component < \{width}; component += blockDim.x) ring[(unsigned long long)slot * \{width}ULL + component] = main_kv[(unsigned long long)(latest - base) * \{width}ULL + component];
      $|}
      $|extern "C" __global__ void \{symbols[2]}(const int32_t *descriptor, int32_t *history) {
      $|  if (threadIdx.x || blockIdx.x || descriptor[3]) return;
      $|  history[0] = descriptor[0] + descriptor[1];
      $|}
    )
  }
  let source = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|#include <math.h>
      $|\{reserve}
      $|\{publication}
      $|extern "C" __global__ void \{symbols[3]}(const int32_t *descriptor, const __nv_bfloat16 *queries, const __nv_bfloat16 *draft_kv, const __nv_bfloat16 *ring, const float *sink, __nv_bfloat16 *output) {
      $|  const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
      $|  if (descriptor[3] || row >= descriptor[2] || head >= \{heads}) return;
      $|  const int main = min(\{window}, descriptor[0] + 1), length = main + descriptor[2];
      $|  const unsigned long long query_base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
      $|  __shared__ float scores[\{window + draft_rows}]; __shared__ float denominator;
      $|  if (lane == 0) {
      $|    float maximum = sink[head];
      $|    for (int slot = 0; slot < length; ++slot) {
      $|      const __nv_bfloat16 *key = slot < main ? ring + (unsigned long long)slot * \{width}ULL : draft_kv + (unsigned long long)(slot - main) * \{width}ULL;
      $|      float dot = 0.0f; for (int component = 0; component < \{width}; ++component) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(queries[query_base + component]), __bfloat162float(key[component])));
      $|      scores[slot] = __fmul_rn(dot, rsqrtf(\{width}.0f)); maximum = fmaxf(maximum, scores[slot]);
      $|    }
      $|    float sum = expf(__fsub_rn(sink[head], maximum));
      $|    for (int slot = 0; slot < length; ++slot) { scores[slot] = expf(__fsub_rn(scores[slot], maximum)); sum = __fadd_rn(sum, scores[slot]); }
      $|    denominator = sum;
      $|  }
      $|  __syncthreads();
      $|  for (int component = lane; component < \{width}; component += blockDim.x) {
      $|    float sum = 0.0f;
      $|    for (int slot = 0; slot < length; ++slot) {
      $|      const __nv_bfloat16 *value = slot < main ? ring + (unsigned long long)slot * \{width}ULL : draft_kv + (unsigned long long)(slot - main) * \{width}ULL;
      $|      sum = __fadd_rn(sum, __fmul_rn(scores[slot], __bfloat162float(value[component])));
      $|    }
      $|    output[query_base + component] = __float2bfloat16_rn(__fdiv_rn(sum, denominator));
      $|  }
      $|}
    ),
  )
  { plan, phase, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn CommittedDraftKvCudaSource::plan(
  self : Self,
) -> @precision.CommittedDraftKvPrecision {
  self.plan
}

///|
pub fn CommittedDraftKvCudaSource::phase(
  self : Self,
) -> @precision.CommittedDraftPhase {
  self.phase
}

///|
pub fn CommittedDraftKvCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn CommittedDraftKvCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[if self.phase is PredictCommitted && stage == 1 {
    3
  } else {
    stage
  }]
}

///|
pub fn CommittedDraftKvCudaSource::stage_count(self : Self) -> Int {
  match self.phase {
    Prime => 3
    Predict => 4
    PredictCommitted => 2
  }
}

///|
pub fn CommittedDraftKvCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, _, window, _, rows, _) = self.plan.geometry()
  match (if self.phase is PredictCommitted && stage == 1 { 3 } else { stage }) {
    1 => (window, 1, 128)
    3 => (rows, heads, 128)
    _ => (1, 1, 1)
  }
}

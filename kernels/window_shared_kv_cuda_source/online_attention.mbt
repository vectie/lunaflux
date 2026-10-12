///|
/// Shared lowering of an explicit online probability law. Model builders own
/// the choice; this renderer knows neither checkpoint families nor token text.
/// Lookup must set key=nullptr for masked logical slots. Keeping those slots
/// preserves block boundaries instead of compacting two independent read sets.
pub fn online_window_attention_source(
  plan : @precision.WindowSharedKvPrecision,
  symbol_prefix~ : String,
  maximum_slots~ : Int,
  logical_slots~ : String,
  setup~ : String,
  lookup~ : String,
) -> Bytes raise {
  if plan.accumulation_law() ==
    @precision.AttentionAccumulationLaw::TiledFloat32 {
    return matrix_window_attention_source(
      plan,
      symbol_prefix~,
      maximum_slots~,
      logical_slots~,
      setup~,
      lookup~,
    )
  }
  guard plan.probability_law() is OnlineBf16(block_tokens~) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (heads, width, window, rows, _) = plan.geometry()
  let blocks = (maximum_slots + block_tokens - 1) / block_tokens
  guard maximum_slots > 0 &&
    (maximum_slots.to_int64() + blocks.to_int64()) * 4L + 4L <= 49152L else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @utf8.encode(
    (
      $|extern "C" __global__ void \{symbol_prefix}_attention(\{setup}) {
      $|  const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
      $|  if (row >= \{rows} || head >= \{heads}) return;
      $|  const unsigned long long query_base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
      $|  if (append[2] || row >= append[1]) { for (int col = lane; col < \{width}; col += blockDim.x) output[query_base + col] = __float2bfloat16_rn(0.0f); return; }
      $|  const int base = append[0], position = base + row;
      $|  const int first = position >= \{window - 1} ? position - \{window - 1} : 0, length = position - first + 1;
      $|  const int window_span = base == 0 ? min(\{window}, append[1]) : \{window};
      $|  const int slots = \{logical_slots}, blocks = (slots + \{block_tokens - 1}) / \{block_tokens};
      $|  __shared__ float scores[\{maximum_slots}], alpha[\{blocks}], denominator;
      $|  if (lane == 0) {
      $|    for (int ordinal = 0; ordinal < slots; ++ordinal) {
      $|      const __nv_bfloat16 *key = nullptr;
      $|\{lookup}
      $|      float dot = 0.0f;
      $|      if (key) for (int col = 0; col < \{width}; ++col) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(queries[query_base + col]), __bfloat162float(key[col])));
      $|      scores[ordinal] = key ? __fmul_rn(dot, rsqrtf(\{width}.0f)) : -INFINITY;
      $|    }
      $|    float maximum = -INFINITY, total = 0.0f;
      $|    for (int block = 0; block < blocks; ++block) {
      $|      const int begin = block * \{block_tokens}, end = min(slots, begin + \{block_tokens});
      $|      float next = maximum;
      $|      for (int ordinal = begin; ordinal < end; ++ordinal) next = fmaxf(next, scores[ordinal]);
      $|      // Entirely masked early blocks must not manufacture exp(-inf- -inf).
      $|      if (next == -INFINITY) { alpha[block] = 1.0f; for (int ordinal = begin; ordinal < end; ++ordinal) scores[ordinal] = 0.0f; continue; }
      $|      const float rescale = maximum == -INFINITY ? 0.0f : expf(__fsub_rn(maximum, next));
      $|      alpha[block] = rescale;
      $|      float sum = 0.0f;
      $|      for (int ordinal = begin; ordinal < end; ++ordinal) {
      $|        const float probability = expf(__fsub_rn(scores[ordinal], next));
      $|        sum = __fadd_rn(sum, probability);
      $|        scores[ordinal] = __bfloat162float(__float2bfloat16_rn(probability));
      $|      }
      $|      total = __fadd_rn(__fmul_rn(total, rescale), sum); maximum = next;
      $|    }
      $|    denominator = __fadd_rn(total, expf(__fsub_rn(sink[head], maximum)));
      $|  }
      $|  __syncthreads();
      $|  for (int col = lane; col < \{width}; col += blockDim.x) {
      $|    float sum = 0.0f;
      $|    for (int block = 0; block < blocks; ++block) {
      $|      sum = __fmul_rn(sum, alpha[block]);
      $|      const int begin = block * \{block_tokens}, end = min(slots, begin + \{block_tokens});
      $|      for (int ordinal = begin; ordinal < end; ++ordinal) {
      $|        const __nv_bfloat16 *key = nullptr;
      $|\{lookup}
      $|        if (key) sum = __fmaf_rn(scores[ordinal], __bfloat162float(key[col]), sum);
      $|      }
      $|    }
      $|    output[query_base + col] = __float2bfloat16_rn(__fdiv_rn(sum, denominator));
      $|  }
      $|}
    ),
  )
}

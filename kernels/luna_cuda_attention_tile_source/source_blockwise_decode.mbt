///|
/// One subgroup per query head, shared immutable K/V per GQA group. Unlike
/// per-key decode, probabilities and PV are formed as a block and the running
/// state is rescaled once per key tile. The alternative law is explicit in
/// the physical program, symbol and recipe; this is not a silent fallback.
fn emit_cuda_blockwise_decode_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  requested_symbol : String,
  maximum_source_bytes : Int,
  fold : @attention_physical_ir.BlockwiseFold,
  effects : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let (query_tile, keys, dimension) = lowering.tile()
  let (_, grid_y, threads, shared_bytes) = lowering.launch()
  let (heads, kv_heads, abi_dimension, page_tokens) = abi.shape()
  let (input_width, _, _) = abi.layout()
  let (max_rows, max_tokens, max_pages) = abi.limits()
  let group = heads / kv_heads
  let stages = if effects.is_pipelined() { 2 } else { 1 }
  let required_shared = (keys * dimension * 4 * stages + group * keys * 4 + 16).to_int64()
  guard !abi.query_metadata &&
    lowering.phase() == Decode &&
    query_tile == 1 &&
    lowering.compute() == CudaGroupedQueryBlockwiseShuffleDot &&
    (keys == 32 || keys == 64) &&
    dimension == abi_dimension &&
    fold.dimensions() == (dimension, keys) &&
    heads % kv_heads == 0 &&
    threads == group * 32 &&
    grid_y == kv_heads &&
    shared_bytes >= required_shared else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let partitions = lowering.key_value_partitions()
  let partitioned = partitions > 1
  let grain = lowering.key_value_partition_grain()
  guard partitions > 0 && grain > 0 && grain % keys == 0 else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let symbol = if partitioned {
    requested_symbol
  } else {
    requested_symbol +
    "_" +
    fold.numeric_law().replace(old="-probability", new="").split("-").join("_")
  }
  guard valid_cuda_symbol(symbol) else {
    raise InvalidCudaAttentionTileSource(InvalidSymbol)
  }
  let transfer = emit_decode_key_value_transfer(
    abi,
    8,
    hoist_page_lookup=lowering.hoists_paged_key_value_page_lookup(),
  )
  let acquire = if effects.is_pipelined() {
    emit_decode_pipeline_iteration(effects)
  } else {
    cuda_grouped_effects(effects, AcquireKeyOperands, transfer, "    ")
  }
  let release = cuda_grouped_effects(effects, ReleaseTileReaders, "", "    ")
  // Double-slot reuse is ordered by the next acquire; only loop exit needs
  // the terminal publication. It must not become a per-tile CTA barrier.
  let complete = if effects.is_pipelined() {
    cuda_grouped_effects(effects, CompleteTileReaders, "", "  ")
  } else {
    ""
  }
  let value_ready = if effects.is_pipelined() {
    cuda_grouped_effects(effects, AcquireValueOperands, "", "    ") +
    "\n" +
    cuda_grouped_effects(effects, PrefetchNextTile, "", "    ")
  } else {
    ""
  }
  let key_released = cuda_grouped_effects(
    effects,
    ReleaseKeyReaders,
    "",
    "    ",
  )
  let body = cuda_blockwise_fold_body(
    fold,
    value_ready~,
    key_released~,
    value_released=if effects.independently_refills() { release } else { "" },
    steps=effects.blockwise_steps(fold),
  )
  let outer_release = if effects.independently_refills() { "" } else { release }
  let initialize = if effects.is_pipelined() {
    emit_decode_async_stage(
      abi,
      effects,
      lowering.hoists_paged_key_value_page_lookup(),
      lowering.decode_copy_retention().unwrap(),
      vector_elements=lowering.decode_copy_elements(),
      tile_rows=keys,
      start_alignment=if partitioned { grain } else { keys },
    )
  } else {
    ""
  }
  let items = (dimension + 31) / 32
  let value_ownership = @attention_physical_ir.ValueComponentOwnership::new(
    dimensions=dimension,
    owners=32,
    maximum_fragment=if dimension == 128 { 4 } else { 1 },
  ).unwrap()
  let value_component = if value_ownership.fragment_width() == 4 {
    "lane * 4 + item"
  } else {
    "lane + item * 32"
  }
  let score_owners = fold.score_owners()
  let score_lanes = 32 / score_owners
  let score_items = (dimension + score_lanes - 1) / score_lanes
  let score_phase = lowering.decode_score_group_phase()
  let score_component = if score_phase != 0 {
    "score_lane + ((item + score_owner * \{score_phase}) % LF_SCORE_QUERY_ITEMS) * LF_SCORE_LANES"
  } else {
    "score_lane + item * LF_SCORE_LANES"
  }
  let output_arguments = if partitioned {
    "const __nv_bfloat16 *rotated_qkv_rows, const __nv_bfloat16 *key_cache,\n    const __nv_bfloat16 *value_cache, float *workspace"
  } else {
    "const __nv_bfloat16 *rotated_qkv_rows, __nv_bfloat16 *output,\n    const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache"
  }
  let state_count = max_rows.to_int64() *
    heads.to_int64() *
    partitions.to_int64()
  let state_initialization = if partitioned {
    "  const unsigned long long state = ((unsigned long long)row * \{heads}ULL + query_head) * \{partitions}ULL + blockIdx.z;\n" +
    "  if (lane == 0) { workspace[state] = CUDART_NAN_F; workspace[\{state_count}ULL + state] = CUDART_NAN_F; }"
  } else {
    ""
  }
  let range = if partitioned {
    "  const int tiles = (query_position + \{grain}) / \{grain};\n" +
    "  const int first = (int)((unsigned long long)tiles * blockIdx.z / \{partitions}ULL);\n" +
    "  const int last = (int)((unsigned long long)tiles * (blockIdx.z + 1) / \{partitions}ULL);\n" +
    "  const int key_begin = first * \{grain};\n  const int key_limit = min(query_position + 1, last * \{grain});"
  } else {
    "  const int key_limit = query_position + 1;\n  const int key_begin = 0;"
  }
  let epilogue = if partitioned {
    "  if (lane == 0) { workspace[state] = maximum; workspace[\{state_count}ULL + state] = denominator; }\n" +
    "  #pragma unroll\n  for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {\n" +
    "    const int component = LF_VALUE_COMPONENT(item);\n" +
    "    if (component < LF_HEAD_DIMENSION) workspace[\{2L * state_count}ULL + state * LF_HEAD_DIMENSION + component] = accumulator[item];\n  }"
  } else {
    "  const unsigned long long output_offset = (unsigned long long)token * \{heads * dimension}ULL + query_head * LF_HEAD_DIMENSION;\n" +
    "  #pragma unroll\n  for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {\n" +
    "    const int component = LF_VALUE_COMPONENT(item);\n    if (component < LF_HEAD_DIMENSION)\n" +
    "      output[output_offset + component] = __float2bfloat16_rn(accumulator[item] / denominator);\n  }"
  }
  let stage_valid = if effects.is_pipelined() {
    "  int *stage_valid = tile_valid;"
  } else {
    ""
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|// numeric_law=\{fold.numeric_law()}
    $|#define LF_BLOCK_THREADS \{threads}
    $|#define LF_KEY_VALUE_TILE \{keys}
    $|#define LF_HEAD_DIMENSION \{dimension}
    $|#define LF_ACCUMULATOR_ITEMS \{items}
    $|#define LF_VALUE_COMPONENT(item) (\{value_component})
    $|#define LF_SCORE_OWNERS \{score_owners}
    $|#define LF_SCORE_LANES \{score_lanes}
    $|#define LF_SCORE_QUERY_ITEMS \{score_items}
    $|#define LF_SCORE_COMPONENT(item) (\{score_component})
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *query_positions,
    $|    const int *query_row_offsets, const int *sequence_lengths,
    $|    const int *page_table_offsets, const int *page_table_page_indices,
    $|    \{output_arguments}) {
    $|  const int thread = (int)threadIdx.x;
    $|  const int lane = thread & 31;
    $|  const int score_lane = lane % LF_SCORE_LANES;
    $|  const int score_owner = lane / LF_SCORE_LANES;
    $|  const int warp = thread >> 5;
    $|  const int key_value_head = (int)blockIdx.y;
    $|  const int query_head = key_value_head * \{group} + warp;
    $|  const int local_row = (int)blockIdx.x;
    $|  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.z >= \{partitions} || key_value_head >= \{kv_heads} || warp >= \{group}) return;
    $|  const int prefill_count = counts[0], decode_count = counts[1];
    $|  const int row_count = counts[2], token_count = counts[3], page_entry_count = counts[4];
    $|  if (prefill_count < 0 || decode_count <= 0 || row_count != prefill_count + decode_count ||
    $|      row_count > \{max_rows} || token_count <= 0 || token_count > \{max_tokens} ||
    $|      page_entry_count <= 0 || page_entry_count > \{max_pages} || local_row >= decode_count) return;
    $|  const int row = prefill_count + local_row;
    $|\{state_initialization}
    $|  const int token = query_row_offsets[row], row_end = query_row_offsets[row + 1];
    $|  const int sequence_length = sequence_lengths[row];
    $|  const int query_position = token >= 0 && token < token_count ? query_positions[token] : -1;
    $|  const int page_start = page_table_offsets[row], page_end = page_table_offsets[row + 1];
    $|  if (token < 0 || row_end != token + 1 || row_end > token_count || query_position < 0 ||
    $|      query_position >= sequence_length || page_start < 0 || page_start >= page_end ||
    $|      page_end > page_entry_count || page_end - page_start !=
    $|          (sequence_length + \{page_tokens - 1}) / \{page_tokens}) return;
    $|  extern __shared__ __align__(16) unsigned char shared_arena[];
    $|  __nv_bfloat16 *stage_arena = reinterpret_cast<__nv_bfloat16 *>(shared_arena);
    $|  __nv_bfloat16 *key_tile = stage_arena;
    $|  __nv_bfloat16 *value_tile = key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|  float *scores = reinterpret_cast<float *>(stage_arena + \{stages * 2} * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION) + warp * LF_KEY_VALUE_TILE;
    $|  int *tile_valid = reinterpret_cast<int *>(stage_arena + \{stages * 2} * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION) + \{group * keys};
    $|\{stage_valid}
    $|  const unsigned long long query_offset = (unsigned long long)token * \{input_width}ULL + query_head * LF_HEAD_DIMENSION;
    $|  float query_fragment[LF_SCORE_QUERY_ITEMS], accumulator[LF_ACCUMULATOR_ITEMS] = {};
    $|  #pragma unroll
    $|  for (int item = 0; item < LF_SCORE_QUERY_ITEMS; ++item) {
    $|    const int component = LF_SCORE_COMPONENT(item);
    $|    query_fragment[item] = component < LF_HEAD_DIMENSION ? __bfloat162float(rotated_qkv_rows[query_offset + component]) : 0.0f;
    $|  }
    $|  const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|  float maximum = -CUDART_INF_F, denominator = 0.0f;
    $|\{range}
    $|\{initialize}
    $|  for (int key_base = key_begin; key_base < key_limit; key_base += LF_KEY_VALUE_TILE) {
    $|    const int active_keys = min(LF_KEY_VALUE_TILE, key_limit - key_base);
    $|\{acquire}
    $|\{body}
    $|\{outer_release}
    $|  }
    $|\{complete}
    $|\{epilogue}
    $|}
    $|#undef LF_BLOCK_THREADS
    $|#undef LF_KEY_VALUE_TILE
    $|#undef LF_HEAD_DIMENSION
    $|#undef LF_ACCUMULATOR_ITEMS
    $|#undef LF_VALUE_COMPONENT
    $|#undef LF_SCORE_OWNERS
    $|#undef LF_SCORE_LANES
    $|#undef LF_SCORE_QUERY_ITEMS
    $|#undef LF_SCORE_COMPONENT
  let bytes = @utf8.encode(source)
  guard bytes.length() <= maximum_source_bytes else {
    raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  {
    lowering_digest: lowering.digest(),
    abi,
    symbol,
    bytes,
    digest: CudaAttentionTileSourceDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    ),
  }
}

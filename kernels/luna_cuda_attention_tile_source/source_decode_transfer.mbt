///|
/// Realize the schedule's contiguous column map as bit-preserving transfers.
/// Width is a schedule input, not a model-name heuristic. A page stride that
/// cannot preserve vector alignment falls back at compile time, not per token.
/// Cache allocations and the shared arena have at least 16-byte alignment.
/// No vector crosses a head or a token/page boundary; inactive keys are zeroed.
fn emit_decode_key_value_transfer(
  abi : CudaPagedAttentionTileAbi,
  vector_bytes : Int,
) -> String {
  let (_, key_value_heads, head_dimension, tokens_per_page) = abi.shape()
  let (_, page_stride_values, physical_page_capacity) = abi.layout()
  let elements = vector_bytes / 2
  let vectorizable = (
      vector_bytes == 4 || vector_bytes == 8 || vector_bytes == 16
    ) &&
    head_dimension % elements == 0 &&
    page_stride_values % elements.to_int64() == 0L
  let (packed_type, width) = if vectorizable {
    (
      match vector_bytes {
        4 => "unsigned int"
        8 => "uint2"
        _ => "uint4"
      },
      elements,
    )
  } else {
    ("__nv_bfloat16", 1)
  }
  let source =
    $|    // Contiguous column map: \{width} BF16 values per transfer.
    $|    for (int vector = thread;
    $|         vector < LF_KEY_VALUE_TILE * (LF_HEAD_DIMENSION / \{width});
    $|         vector += LF_BLOCK_THREADS) {
    $|      const int linear = vector * \{width};
    $|      const int key = linear / LF_HEAD_DIMENSION;
    $|      const int component = linear % LF_HEAD_DIMENSION;
    $|      \{packed_type} staged_key = {};
    $|      \{packed_type} staged_value = {};
    $|      if (key < active_keys) {
    $|        const int key_position = key_base + key;
    $|        const int page_slot = page_start + key_position / \{tokens_per_page};
    $|        const int physical_page =
    $|            page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (physical_page < 0 ||
    $|            (unsigned long long)physical_page >= \{physical_page_capacity}ULL) {
    $|          atomicExch(tile_valid, 0);
    $|        } else {
    $|          const int token_in_page = key_position % \{tokens_per_page};
    $|          const unsigned long long cache_index =
    $|              (unsigned long long)physical_page * \{page_stride_values}ULL +
    $|              ((unsigned long long)token_in_page * \{key_value_heads}ULL +
    $|                  key_value_head) * LF_HEAD_DIMENSION + component;
    $|          staged_key = *reinterpret_cast<const \{packed_type} *>(key_cache + cache_index);
    $|          staged_value = *reinterpret_cast<const \{packed_type} *>(value_cache + cache_index);
    $|        }
    $|      }
    $|      *reinterpret_cast<\{packed_type} *>(key_tile + linear) = staged_key;
    $|      *reinterpret_cast<\{packed_type} *>(value_tile + linear) = staged_value;
    $|    }
  source
}

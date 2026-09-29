///|
/// Realize the schedule's contiguous column map as bit-preserving transfers.
/// Width is a schedule input, not a model-name heuristic. A page stride that
/// cannot preserve vector alignment falls back at compile time, not per token.
/// Cache allocations and the shared arena have at least 16-byte alignment.
/// No vector crosses a head or a token/page boundary; inactive keys are zeroed.
fn emit_decode_key_value_transfer(
  abi : CudaPagedAttentionTileAbi,
  vector_bytes : Int,
  hoist_page_lookup? : Bool = false,
) -> String {
  let (_, _, head_dimension, _) = abi.shape()
  let (_, page_stride_values, _) = abi.layout()
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
  let address = cuda_decode_row_address(
    abi, width, hoist_page_lookup, "atomicExch(tile_valid, 0)",
  )
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
    $|\{address}
    $|      if (row_valid) {
    $|          staged_key = *reinterpret_cast<const \{packed_type} *>(key_cache + cache_index);
    $|          staged_value = *reinterpret_cast<const \{packed_type} *>(value_cache + cache_index);
    $|      }
    $|      *reinterpret_cast<\{packed_type} *>(key_tile + linear) = staged_key;
    $|      *reinterpret_cast<\{packed_type} *>(value_tile + linear) = staged_value;
    $|    }
  source
}

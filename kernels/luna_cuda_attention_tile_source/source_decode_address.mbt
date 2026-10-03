///|
/// Realize the retained page-lookup rewrite for both cooperative and async
/// grouped copies. Distribution is outside the active-key branch: even lanes
/// zero-filling a partial tile participate in the subgroup collective.
fn cuda_decode_row_address(
  abi : CudaPagedAttentionTileAbi,
  vector_elements : Int,
  hoist : Bool,
  invalid : String,
  retained_page? : String = "",
) -> String {
  let (_, heads, dimension, page_tokens) = abi.shape()
  let (_, stride, capacity) = abi.layout()
  let ownership = @attention_physical_ir.PagedRowOwnership::new(
    row_elements=dimension,
    vector_elements~,
    subgroup_size=32,
    hoist~,
  ).unwrap()
  let leader = if ownership.shares_address() {
    "component == 0"
  } else {
    "true"
  }
  if !retained_page.is_empty() {
    // The page-epoch plan retains a narrow page identity, not a 64-bit
    // address. Reconstruct the immutable affine offset at each consumer;
    // distributing the address again costs two shuffles plus a validity
    // shuffle after the already-distributed page identity.
    return (
      $|\{retained_page}
      $|      const int position = key_base + key;
      $|      const int page = retained_page;
      $|      const int row_valid = key < active_keys && page >= 0 && (unsigned long long)page < \{capacity}ULL;
      $|      if (key < active_keys && !row_valid && \{leader}) { \{invalid}; }
      $|      const unsigned long long row_address = row_valid
      $|          ? (unsigned long long)page * \{stride}ULL +
      $|              ((unsigned long long)(position % \{page_tokens}) * \{heads}ULL + key_value_head) * LF_HEAD_DIMENSION
      $|          : 0;
      $|      const unsigned long long cache_index = row_valid ? row_address + component : 0;
    )
  }
  let distribute = if ownership.shares_address() {
    "      row_address = __shfl_sync(0xffffffffU, row_address, (lane / \{ownership.consumers()}) * \{ownership.consumers()});\n" +
    "      row_valid = __shfl_sync(0xffffffffU, row_valid, (lane / \{ownership.consumers()}) * \{ownership.consumers()});"
  } else {
    ""
  }
  (
    $|      unsigned long long row_address = 0;
    $|      int row_valid = 0;
    $|      if (key < active_keys && \{leader}) {
    $|        const int position = key_base + key;
    $|        const int page_slot = page_start + position / \{page_tokens};
    $|        const int page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (page < 0 || (unsigned long long)page >= \{capacity}ULL) {
    $|          \{invalid};
    $|        } else {
    $|          row_address = (unsigned long long)page * \{stride}ULL +
    $|              ((unsigned long long)(position % \{page_tokens}) * \{heads}ULL + key_value_head) * LF_HEAD_DIMENSION;
    $|          row_valid = 1;
    $|        }
    $|      }
    $|\{distribute}
    $|      const unsigned long long cache_index = row_valid ? row_address + component : 0;
  )
}

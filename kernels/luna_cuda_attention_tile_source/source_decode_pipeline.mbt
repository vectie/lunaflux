///|
/// Two-stage realization of the schedule's asynchronous shared-memory map.
/// Only storage lifetimes change: the ordered softmax fold remains untouched.
fn emit_decode_async_stage(abi : CudaPagedAttentionTileAbi) -> String {
  let (_, heads, _, page_tokens) = abi.shape()
  let (_, stride, capacity) = abi.layout()
  (
    $|  auto stage_tile = [&](int key_base, int slot) {
    $|    const int active_keys = min(LF_KEY_VALUE_TILE, key_limit - key_base);
    $|    __nv_bfloat16 *key_tile = stage_arena + slot * 2 * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    __nv_bfloat16 *value_tile = key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    constexpr int copies = (LF_KEY_VALUE_TILE * (LF_HEAD_DIMENSION / 4) + LF_BLOCK_THREADS - 1) / LF_BLOCK_THREADS;
    $|    unsigned long long addresses[copies];
    $|    int sizes[copies];
    $|    int item = 0;
    $|    for (int vector = thread; vector < LF_KEY_VALUE_TILE * (LF_HEAD_DIMENSION / 4);
    $|         vector += LF_BLOCK_THREADS) {
    $|      const int linear = vector * 4;
    $|      const int key = linear / LF_HEAD_DIMENSION;
    $|      const int component = linear % LF_HEAD_DIMENSION;
    $|      unsigned long long cache_index = 0;
    $|      int source_bytes = 0;
    $|      if (key < active_keys) {
    $|        const int position = key_base + key;
    $|        const int page_slot = page_start + position / \{page_tokens};
    $|        const int page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (page < 0 || (unsigned long long)page >= \{capacity}ULL) {
    $|          atomicExch(stage_valid + slot, 0);
    $|        } else {
    $|          cache_index = (unsigned long long)page * \{stride}ULL +
    $|              ((unsigned long long)(position % \{page_tokens}) * \{heads}ULL + key_value_head) *
    $|                  LF_HEAD_DIMENSION + component;
    $|          source_bytes = 8;
    $|        }
    $|      }
    $|      const unsigned int key_address = (unsigned int)__cvta_generic_to_shared(key_tile + linear);
    $|      addresses[item] = cache_index;
    $|      sizes[item++] = source_bytes;
    $|      asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
    $|          "r"(key_address), "l"(key_cache + cache_index), "r"(source_bytes) : "memory");
    $|    }
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|    item = 0;
    $|    for (int vector = thread; vector < LF_KEY_VALUE_TILE * (LF_HEAD_DIMENSION / 4);
    $|         vector += LF_BLOCK_THREADS) {
    $|      const unsigned int value_address = (unsigned int)__cvta_generic_to_shared(value_tile + vector * 4);
    $|      asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
    $|          "r"(value_address), "l"(value_cache + addresses[item]), "r"(sizes[item]) : "memory");
    $|      ++item;
    $|    }
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|  };
    $|  if (thread == 0) { stage_valid[0] = 1; stage_valid[1] = 1; }
    $|  __syncthreads();
    $|  if (key_begin < key_limit) stage_tile(key_begin, 0);
  )
}

///|
fn emit_decode_pipeline_iteration() -> String {
  (
    $|    const int slot = ((key_base - key_begin) / LF_KEY_VALUE_TILE) & 1;
    $|    // K is ready before V: QK does not depend on the value transfer.
    $|    asm volatile("cp.async.wait_group 1;" ::: "memory");
    $|    __syncthreads();
    $|    key_tile = stage_arena + slot * 2 * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    value_tile = key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
  )
}

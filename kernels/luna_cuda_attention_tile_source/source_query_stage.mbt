///|
/// One immutable KV read map, specialized to the schedule's transfer primitive.
/// Future transfers own a disjoint ring slot; the final consumer barrier ends
/// its previous lifetime before that slot may be reused.
fn cuda_query_owned_stage(
  abi : CudaPagedAttentionTileAbi,
  dense : Bool,
  asynchronous : Bool,
) -> String {
  let transfer = if asynchronous {
    let text =
      #|        const int bytes=kv_source_valid?16:0;
      #|        const unsigned ka=(unsigned)__cvta_generic_to_shared(destination_k);
      #|        const unsigned va=(unsigned)__cvta_generic_to_shared(destination_v);
      #|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(ka),"l"(kv_key_source),"r"(bytes):"memory");
      #|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(va),"l"(kv_value_source),"r"(bytes):"memory");
    text
  } else {
    let text =
      #|        uint4 pk=make_uint4(0,0,0,0), pv=pk;
      #|        if (kv_source_valid) {
      #|          pk=*reinterpret_cast<const uint4 *>(kv_key_source);
      #|          pv=*reinterpret_cast<const uint4 *>(kv_value_source);
      #|        }
      #|        *reinterpret_cast<uint4 *>(destination_k)=pk;
      #|        *reinterpret_cast<uint4 *>(destination_v)=pv;
    text
  }
  let commit = if asynchronous {
    "asm volatile(\"cp.async.commit_group;\" ::: \"memory\");"
  } else {
    ""
  }
  let source =
    $|    auto stage_query_tile = [&](int kb, int stage) {
    $|      for (int vector=thread; vector<LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION/8; vector+=LF_BLOCK_THREADS) {
    $|        const int r=vector*8/LF_HEAD_DIMENSION, c=vector*8%LF_HEAD_DIMENSION;
    $|        const int position=kb+r;
    $|\{cuda_prefill_key_value_view_address(abi, true, dense~)}
    $|        auto *destination_k=kbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|        auto *destination_v=vbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|\{transfer}
    $|      }
    $|      \{commit}
    $|    };
  source
}

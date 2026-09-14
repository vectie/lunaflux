///|
/// One immutable KV read map, specialized to the schedule's transfer primitive.
/// Future transfers own a disjoint ring slot; the final consumer barrier ends
/// its previous lifetime before that slot may be reused.
fn cuda_query_owned_stage(
  abi : CudaPagedAttentionTileAbi,
  dense : Bool,
  asynchronous : Bool,
  split? : Bool = false,
) -> String {
  let transfer = if asynchronous {
    let text =
      $|        const int bytes=kv_source_valid?16:0;
      $|        const unsigned ka=(unsigned)__cvta_generic_to_shared(destination_k);
      $|        const unsigned va=(unsigned)__cvta_generic_to_shared(destination_v);
      $|        if (\{if split { "kind==0" } else { "true" }}) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(ka),"l"(kv_key_source),"r"(bytes):"memory");
      $|        if (\{if split { "kind==1" } else { "true" }}) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(va),"l"(kv_value_source),"r"(bytes):"memory");
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
  let retained = if split {
    // K produces the immutable address product; V consumes it before the
    // next K transfer overwrites it. Fully unroll to scalarize the product.
    (
      #|    const __nv_bfloat16 *pending_values[LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION/8/LF_BLOCK_THREADS];
      #|    const int row_position_begin=positions[begin];
    )
  } else {
    ""
  }
  let vectors = if split {
    (
      #|      #pragma unroll
      #|      for (int index=0; index<LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION/8/LF_BLOCK_THREADS; ++index) {
      #|        const int vector=thread+index*LF_BLOCK_THREADS;
    )
  } else {
    (
      #|      for (int vector=thread; vector<LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION/8; vector+=LF_BLOCK_THREADS) {
    )
  }
  // Partial evaluation of the interval union: a whole historical tile cannot
  // intersect the current rows. Keep the mixed edge tile and the retained V
  // consumer unchanged; only K produces a new address product.
  let specialize = split && dense
  let mixed = cuda_query_owned_stage_body(
    abi,
    dense,
    split,
    if specialize {
      "stage_mixed_tile"
    } else {
      "stage_query_tile"
    },
    vectors,
    transfer,
    commit,
  )
  if specialize {
    retained +
    mixed +
    cuda_query_owned_stage_body(
      abi, false, split, "stage_history_tile", vectors, transfer, commit,
    ) +
    (
      #|    auto stage_query_tile = [&](int kb, int stage, int kind) {
      #|      if (kind==0 && (long long)kb+LF_KEY_VALUE_TILE<=row_position_begin)
      #|        stage_history_tile(kb,stage,kind);
      #|      else stage_mixed_tile(kb,stage,kind);
      #|    };
    )
  } else {
    retained + mixed
  }
}

///|
fn cuda_query_owned_stage_body(
  abi : CudaPagedAttentionTileAbi,
  dense : Bool,
  split : Bool,
  name : String,
  vectors : String,
  transfer : String,
  commit : String,
) -> String {
  let source =
    $|    auto \{name} = [&](int kb, int stage\{if split { ", int kind" } else { "" }}) {
    $|\{vectors}
    $|        const int r=vector*8/LF_HEAD_DIMENSION, c=vector*8%LF_HEAD_DIMENSION;
    $|        const int position=kb+r;
    $|\{cuda_prefill_key_value_view_address(abi, true, dense~, retained_value=split)}
    $|        auto *destination_k=kbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|        auto *destination_v=vbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|\{transfer}
    $|      }
    $|      \{commit}
    $|    };
  source
}

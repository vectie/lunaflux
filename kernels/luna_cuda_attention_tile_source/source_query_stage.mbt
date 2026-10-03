///|
/// One immutable KV read map, specialized to the schedule's transfer primitive.
/// Future transfers own a disjoint ring slot; the final consumer barrier ends
/// its previous lifetime before that slot may be reused.
fn cuda_query_owned_stage(
  abi : CudaPagedAttentionTileAbi,
  dense : Bool,
  asynchronous : Bool,
  split? : Bool = false,
  affine_positions? : Bool = false,
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
    affine_positions~,
  )
  if specialize {
    let current = if affine_positions {
      cuda_query_current_stage(abi, vectors, transfer, commit)
    } else {
      ""
    }
    let current_dispatch = if affine_positions {
      "      if (kind==0 && kb>=row_position_begin && (long long)kb+LF_KEY_VALUE_TILE<=row_position_begin+(end-begin) && (long long)kb+LF_KEY_VALUE_TILE<=key_end) stage_current_tile(kb,stage);\n      else "
    } else {
      "      "
    }
    let historical = cuda_query_complete_history_stage(
      abi, vectors, transfer, commit,
    )
    let page_tokens = abi.shape().3
    retained +
    mixed +
    cuda_query_owned_stage_body(
      abi, false, split, "stage_history_tile", vectors, transfer, commit,
    ) +
    current +
    historical +
    (
      $|    auto stage_query_tile = [&](int kb, int stage, int kind) {
      $|\{current_dispatch}if (kind==0 && kb%\{page_tokens}==0 && (long long)kb+LF_KEY_VALUE_TILE<=min(row_position_begin,key_end))
      $|        stage_complete_history_tile(kb,stage);
      $|      else if (kind==0 && (long long)kb+LF_KEY_VALUE_TILE<=row_position_begin)
      $|        stage_history_tile(kb,stage,kind);
      $|      else stage_mixed_tile(kb,stage,kind);
      $|    };
    )
  } else {
    retained + mixed
  }
}

///|
/// On a complete page-aligned historical interval, subgroup lanes share the
/// same immutable page lookup. Derive the sharing width from the ABI, not a
/// model or device shape. Validate each page at its owner before broadcasting;
/// retain V addresses and preserve the existing invalid-page publication.
fn cuda_query_complete_history_stage(
  abi : CudaPagedAttentionTileAbi,
  vectors : String,
  transfer : String,
  commit : String,
) -> String {
  let (heads, kv_heads, dimension, page_tokens) = abi.shape()
  let (_, page_stride, pages) = abi.layout()
  let lanes_per_page = (page_tokens * (dimension / 8)).min(32)
  (
    $|    auto stage_complete_history_tile = [&](int kb, int stage) {
    $|\{vectors}
    $|        const int r=vector*8/LF_HEAD_DIMENSION, c=vector*8%LF_HEAD_DIMENSION;
    $|        const int position=kb+r;
    $|        int kv_page=-1;
    $|        if (lane%\{lanes_per_page}==0) {
    $|          kv_page=page_indices[page_begin+position/\{page_tokens}];
    $|          if (kv_page<0 || (unsigned long long)kv_page>=\{pages}ULL) { atomicExch(control,0); kv_page=-1; }
    $|        }
    $|        kv_page=__shfl_sync(0xffffffffU,kv_page,(lane/\{lanes_per_page})*\{lanes_per_page});
    $|        const bool kv_source_valid=kv_page>=0;
    $|        const unsigned long long source=kv_source_valid?(unsigned long long)kv_page*\{page_stride}ULL+((unsigned long long)(position%\{page_tokens})*\{kv_heads}ULL+head/\{heads/kv_heads})*LF_HEAD_DIMENSION+c:0ULL;
    $|        const __nv_bfloat16 *kv_key_source=keys+source, *kv_value_source=values+source;
    $|        pending_values[index]=kv_source_valid?kv_value_source:nullptr;
    $|        const int kind=0;
    $|        auto *destination_k=kbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|        auto *destination_v=vbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|\{transfer}
    $|      }
    $|      \{commit}
    $|    };
  )
}

///|
/// Refine a complete affine current interval before emitting its vector map.
/// The retained read-view proves equivalence to committed KV. No page lookup,
/// masked transfer or per-vector interval test is needed in this subdomain;
/// mixed and historical intervals keep the original checked read map.
fn cuda_query_current_stage(
  abi : CudaPagedAttentionTileAbi,
  vectors : String,
  transfer : String,
  commit : String,
) -> String {
  let (heads, kv_heads, _, _) = abi.shape()
  let (row_width, _, _) = abi.layout()
  (
    $|    auto stage_current_tile = [&](int kb, int stage) {
    $|\{vectors}
    $|        const int r=vector*8/LF_HEAD_DIMENSION, c=vector*8%LF_HEAD_DIMENSION;
    $|        const unsigned long long source=(unsigned long long)(begin+kb-row_position_begin+r)*\{row_width}ULL+(\{heads}ULL+head/\{heads/kv_heads})*LF_HEAD_DIMENSION+c;
    $|        const __nv_bfloat16 *kv_key_source=input+source;
    $|        const __nv_bfloat16 *kv_value_source=kv_key_source+\{kv_heads}*LF_HEAD_DIMENSION;
    $|        const bool kv_source_valid=true;
    $|        const int kind=0;
    $|        pending_values[index]=kv_value_source;
    $|        auto *destination_k=kbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|        auto *destination_v=vbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|\{transfer}
    $|      }
    $|      \{commit}
    $|    };
  )
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
  affine_positions? : Bool = false,
) -> String {
  let source =
    $|    auto \{name} = [&](int kb, int stage\{if split { ", int kind" } else { "" }}) {
    $|\{vectors}
    $|        const int r=vector*8/LF_HEAD_DIMENSION, c=vector*8%LF_HEAD_DIMENSION;
    $|        const int position=kb+r;
    $|\{cuda_prefill_key_value_view_address(abi, true, dense~, retained_value=split, affine_positions~)}
    $|        auto *destination_k=kbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|        auto *destination_v=vbuf+stage*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION+lf_operand_index(r,c,LF_HEAD_DIMENSION);
    $|\{transfer}
    $|      }
    $|      \{commit}
    $|    };
  source
}

///|
/// CUDA instruction realization of the shared column, transfer and ring IRs.
/// The epilogue only receives the completed projection; it cannot change the
/// ordered dot fold or the publication/lifetime of its input operands.
fn ingress_matrix_projection_source(
  physical : QwenIngressPhysicalProgram,
) -> String {
  let pipeline = physical.pipeline.schedule
  let width = pipeline.transfer_width()
  let storage = physical.pipeline.operands
  let input_region = storage.region(0).unwrap()
  let weight_region = storage.region(1).unwrap()
  let (input_elements, _, _) = weight_region.geometry()
  let slot_elements = storage.slot_elements()
  let body = @luna_cuda_projection_aot.CudaProjectionFoldBody::ingress(
    physical.pipeline,
  ).unwrap()
  let lifetime = @luna_cuda_projection_aot.lower_projection_fold_lifetime(
    physical.pipeline,
    body,
  )
  let input_vectors = input_region.ownership().geometry().0
  let weight_vectors = weight_region.ownership().geometry().0
  let row_fragments = physical.pipeline.fragments.geometry().0
  let (domain_rows, row_quantum) = physical.pipeline.row_domain.geometry()
  let declaration = if physical.scratch.reuses_storage() ||
    physical.scratch.allocation_bytes() > 49152L {
    "    __nv_bfloat16 *ingress_operands = reinterpret_cast<__nv_bfloat16 *>(ingress_scratch);"
  } else {
    "    __shared__ __align__(32) __nv_bfloat16 ingress_operands[\{storage.allocation_elements()}];"
  }
  let retire = if physical.scratch.reuses_storage() {
    "      // Retire every operand reader before reusing its storage.\n      __syncthreads();"
  } else {
    ""
  }
  // Specialize the complete lifetime once, rather than selecting MMA fan-out
  // inside every reduction iteration. Each branch consumes the same prefix
  // domain for staging, fragment products and publication.
  let dispatch = Array::makei(domain_rows, index => {
    let active = domain_rows - index
    "\{if index == 0 { "if" } else { "else if" }} (live_row_fragments == \{active}) project_live_domain(std::integral_constant<int, \{active}>{});"
  }).join("\n    ")
  let source =
    $|\{declaration}
    $|    const int live_row_fragments = min(\{domain_rows}, 1 + (tokens - token_begin - 1) / \{row_quantum});
    $|    auto project_live_domain = [&](auto live_row_domain) {
    $|    constexpr int ingress_active_fragments = decltype(live_row_domain)::value;
    $|    auto stage_operands = [&](int offset, int stage) {
    $|      __nv_bfloat16 *destination = ingress_operands + stage * \{slot_elements};
    $|      for (int vector = (int)threadIdx.x; vector < min(\{input_vectors}, ingress_active_fragments * \{row_quantum * width / 8}); vector += LF_BLOCK_THREADS) {
    $|        const int local_row = vector / \{width / 8}, local_inner = (vector % \{width / 8}) * 8;
    $|        const bool valid = token_begin + local_row < tokens;
    $|        const __nv_bfloat16 *source = valid ? input + (unsigned long long)(token_begin + local_row) * LF_INPUT_WIDTH + offset + local_inner : input;
    $|        const unsigned address = (unsigned)__cvta_generic_to_shared(destination + vector * 8);
    $|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(valid ? 16 : 0) : "memory");
    $|      }
    $|      for (int vector = (int)threadIdx.x; vector < \{weight_vectors}; vector += LF_BLOCK_THREADS) {
    $|        const int local_column = vector / \{width / 8}, local_inner = (vector % \{width / 8}) * 8;
    $|        const int weight_head = head_begin + local_column / LF_HEAD_DIM;
    $|        const bool valid = weight_head < LF_PACKED_HEADS;
    $|        const bool query_weight = weight_head < LF_QUERY_HEADS, key_weight = weight_head < LF_QK_HEADS;
    $|        const int matrix_head = query_weight ? weight_head : (key_weight ? weight_head - LF_QUERY_HEADS : weight_head - LF_QK_HEADS);
    $|        const __nv_bfloat16 *matrix = query_weight ? q_weight : (key_weight ? k_weight : v_weight);
    $|        const __nv_bfloat16 *source = valid ? matrix + (unsigned long long)(matrix_head * LF_HEAD_DIM + local_column % LF_HEAD_DIM) * LF_INPUT_WIDTH + offset + local_inner : v_weight;
    $|        const unsigned address = (unsigned)__cvta_generic_to_shared(destination + \{input_elements} + vector * 8);
    $|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(valid ? 16 : 0) : "memory");
    $|      }
    $|      asm volatile("cp.async.commit_group;" ::: "memory");
    $|    };
    $|    for (int window = 0; window < LF_COLUMN_WINDOWS; ++window) {
    $|      float c[\{row_fragments}][LF_LIVE_COLUMNS][8] = {};
    $|      \{lifetime}
    $|\{retire}
    $|      #pragma unroll
    $|      for (int slot = 0; slot < LF_LIVE_COLUMNS; ++slot) {
    $|        const int column_base = ((window * LF_LIVE_COLUMNS + slot) * LF_WARPS + warp) * 16;
    $|        #pragma unroll
    $|        for (int row_fragment=0; row_fragment<\{row_fragments}; ++row_fragment)
    $|          if (column_base < LF_PROJECTION_WIDTH && row_fragment < ingress_active_fragments) {
    $|            #pragma unroll
    $|            for (int item=0;item<8;++item) {
    $|              const int row=row_fragment*16+lane/4+(item%4/2)*8;
    $|              const int column=column_base+lane%4*2+item%2+item/4*8;
    $|              projected[row][column]=c[row_fragment][slot][item];
    $|            }
    $|          }
    $|      }
    $|    }
    $|    };
    $|    \{dispatch}
    $|
  source
}

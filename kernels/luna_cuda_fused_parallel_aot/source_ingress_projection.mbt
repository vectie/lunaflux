///|
/// CUDA instruction realization of the shared column, transfer and ring IRs.
/// The epilogue only receives the completed projection; it cannot change the
/// ordered dot fold or the publication/lifetime of its input operands.
fn ingress_matrix_projection_source(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> String {
  let columns = qwen_ingress_column_plan(compilation)
  let pipeline = compilation.schedule().matrix_pipeline().unwrap()
  let (groups, _, live, _) = columns.geometry()
  let rows = pipeline.row_width()
  let width = pipeline.transfer_width()
  let head_width = pipeline.output_tiles() * 16
  let transfers = @luna_projection_tile_compiler.ProjectionOperandTransferPlan::new(
    input_rows=rows,
    weight_rows=head_width,
    row_width=width,
    vector_width=8,
    owners=groups * 32,
  ).unwrap()
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=width,
    vector=8,
    rotations=1,
    row_group=1,
  ).unwrap()
  let storage = transfers
    .storage(layout, stages=pipeline.stage_count())
    .unwrap()
  let input_region = storage.region(0).unwrap()
  let weight_region = storage.region(1).unwrap()
  let (input_elements, _, _) = weight_region.geometry()
  let slot_elements = storage.slot_elements()
  let body = @luna_cuda_projection_aot.CudaProjectionFoldBody::ingress(
    pipeline,
    storage,
    columns=live,
  ).unwrap()
  let lifetime = @luna_cuda_projection_aot.lower_projection_fold_lifetime(
    pipeline,
    body,
    OverlappedTransfer,
  )
  let input_vectors = input_region.ownership().geometry().0
  let weight_vectors = weight_region.ownership().geometry().0
  let source =
    $|    __shared__ __align__(32) __nv_bfloat16 ingress_operands[\{storage.allocation_elements()}];
    $|    auto stage_operands = [&](int offset, int stage) {
    $|      __nv_bfloat16 *destination = ingress_operands + stage * \{slot_elements};
    $|      for (int vector = (int)threadIdx.x; vector < \{input_vectors}; vector += LF_BLOCK_THREADS) {
    $|        const int local_row = vector / \{width / 8}, local_inner = (vector % \{width / 8}) * 8;
    $|        const bool valid = token_begin + local_row < tokens;
    $|        const __nv_bfloat16 *source = valid ? input + (unsigned long long)(token_begin + local_row) * LF_INPUT_WIDTH + offset + local_inner : input;
    $|        const unsigned address = (unsigned)__cvta_generic_to_shared(destination + vector * 8);
    $|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(valid ? 16 : 0) : "memory");
    $|      }
    $|      for (int vector = (int)threadIdx.x; vector < \{weight_vectors}; vector += LF_BLOCK_THREADS) {
    $|        const int local_column = vector / \{width / 8}, local_inner = (vector % \{width / 8}) * 8;
    $|        const __nv_bfloat16 *source = matrix + (unsigned long long)(matrix_head * LF_HEAD_DIM + local_column) * LF_INPUT_WIDTH + offset + local_inner;
    $|        const unsigned address = (unsigned)__cvta_generic_to_shared(destination + \{input_elements} + vector * 8);
    $|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(source) : "memory");
    $|      }
    $|      asm volatile("cp.async.commit_group;" ::: "memory");
    $|    };
    $|    for (int window = 0; window < LF_COLUMN_WINDOWS; ++window) {
    $|      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c[LF_LIVE_COLUMNS];
    $|      #pragma unroll
    $|      for (int slot = 0; slot < LF_LIVE_COLUMNS; ++slot) nvcuda::wmma::fill_fragment(c[slot], 0.0f);
    $|      \{lifetime}
    $|      #pragma unroll
    $|      for (int slot = 0; slot < LF_LIVE_COLUMNS; ++slot) {
    $|        const int column_base = ((window * LF_LIVE_COLUMNS + slot) * LF_WARPS + warp) * 16;
    $|        if (column_base < LF_HEAD_DIM) nvcuda::wmma::store_matrix_sync(&projected[0][column_base], c[slot], LF_HEAD_DIM, nvcuda::wmma::mem_row_major);
    $|      }
    $|    }
    $|
  source
}

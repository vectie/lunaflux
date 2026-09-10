///|
/// A product of ordered folds shares both the staged input and each pair of
/// weight tiles. Copy scheduling and CUDA lane ownership are lowering details.
fn projection_sibling_fold_pipeline_source(
  plan : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  parameters : ProjectionParameters,
  groups~ : Int,
) -> String {
  if @luna_projection_tile_compiler.partition_projection_sibling_reuse(
      row_tiles=plan.row_width() / 16,
      output_tiles=plan.output_tiles(),
      consumer_groups=groups,
    )
    is Some(reuse) {
    return projection_sibling_reuse_source(plan, parameters, reuse, groups~)
  }
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let width = plan.transfer_width()
  let input_width = parameters.input_width
  let intermediate = parameters.intermediate_width
  let slot = (rows + 2 * columns) * width
  let distribution = @luna_projection_tile_compiler.partition_projection_product_fold(
    row_tiles=rows / 16,
    output_tiles=plan.output_tiles(),
    consumer_groups=groups,
  ).unwrap()
  let fold_rows = distribution.rows_per_group()
  let consumers = distribution.consumer_groups()
  let layout : CudaProjectionTransportLayout = { columns: width }
  let copies = StringBuilder()
  for
    domain in [
      (
        "input + (unsigned long long)row_begin * \{input_width}ULL",
        "destination",
        rows,
        true,
      ),
      (
        "gate_weight + (unsigned long long)column_begin * \{input_width}ULL",
        "destination + \{rows * width}",
        columns,
        false,
      ),
      (
        "up_weight + (unsigned long long)column_begin * \{input_width}ULL",
        "destination + \{(rows + columns) * width}",
        columns,
        false,
      ),
    ] {
    let (base, destination, domain_rows, guard_rows) = domain
    // Priming has no previous fold to overlap. Keep its masked input value
    // local to the producer rather than issuing partially active bypass-copy
    // transactions. Both calls below carry constant phase values, so phase
    // selection disappears before the device loop is emitted.
    if guard_rows {
      copies.write_string("if (!overlap) {\n")
      copies.write_string(
        projection_fold_operand_copy_source(
          "(" + base + ")",
          destination,
          domain_rows,
          width,
          input_width,
          guard_rows,
          false,
        ),
      )
      copies.write_string("\n} else {\n")
    }
    copies.write_string(
      projection_fold_operand_copy_source(
        "(" + base + ")",
        destination,
        domain_rows,
        width,
        input_width,
        guard_rows,
        true,
      ),
    )
    if guard_rows {
      copies.write_string("\n}\n")
    }
    copies.write_string("\n")
  }
  let source =
    $|  if (token_count >= \{plan.minimum_rows()}) {
    $|    __shared__ __align__(32) __nv_bfloat16 sibling_storage[\{slot * 2}];
    $|    \{layout.address_function()}
    $|    const int groups = \{intermediate / columns};
    $|    const int work_count = ((token_count + \{rows - 1}) / \{rows}) * groups;
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / groups) * \{rows};
    $|      const int column_begin = (work % groups) * \{columns};
    $|      const int consumer_row = (warp / \{distribution.output_groups()}) * \{fold_rows};
    $|      const int consumer_column = warp % \{distribution.output_groups()};
    $|      auto stage_operands = [&](int k_base, int stage, bool overlap) {
    $|        __nv_bfloat16 *destination = sibling_storage + stage * \{slot};
    $|        const int row_start = row_begin, offset = k_base;
    $|\{copies.to_string()}
    $|        asm volatile("cp.async.commit_group;" ::: "memory");
    $|      };
    $|      float cg[\{fold_rows}][8] = {}, cu[\{fold_rows}][8] = {};
    $|      stage_operands(0, 0, false);
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{input_width}; inner += \{width}) {
    $|        if (inner + \{width} < \{input_width}) stage_operands(inner + \{width}, stage ^ 1, true);
    $|        if (warp < \{consumers}) {
    $|          const __nv_bfloat16 *current = sibling_storage + stage * \{slot};
    $|          #pragma unroll
    $|          for (int k = 0; k < \{width}; k += 16) {
    $|            unsigned bg[4], bu[4];
    $|            const int r = consumer_column * 16 + lane % 16;
    $|            const int component = k + lane / 16 * 8;
    $|            const unsigned gp = (unsigned)__cvta_generic_to_shared(current + \{rows * width} + operand_index(r, component));
    $|            const unsigned up = (unsigned)__cvta_generic_to_shared(current + \{(rows + columns) * width} + operand_index(r, component));
    $|            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(bg[0]), "=r"(bg[1]), "=r"(bg[2]), "=r"(bg[3]) : "r"(gp));
    $|            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(bu[0]), "=r"(bu[1]), "=r"(bu[2]), "=r"(bu[3]) : "r"(up));
    $|            #pragma unroll
    $|            for (int row = 0; row < \{fold_rows}; ++row) {
    $|              if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|              unsigned a[4];
    $|              const unsigned ap = (unsigned)__cvta_generic_to_shared(current + operand_index((consumer_row + row) * 16 + lane % 16, k + lane / 16 * 8));
    $|              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(ap));
    $|              #pragma unroll
    $|              for (int half = 0; half < 2; ++half) {
    $|                float *g = cg[row] + half * 4, *u = cu[row] + half * 4;
    $|                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(g[0]), "+f"(g[1]), "+f"(g[2]), "+f"(g[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bg[half]), "r"(bg[half+2]));
    $|                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(u[0]), "+f"(u[1]), "+f"(u[2]), "+f"(u[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bu[half]), "r"(bu[half+2]));
    $|              }
    $|            }
    $|          }
    $|        }
    $|        asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|        __syncthreads();
    $|        stage ^= 1;
    $|      }
    $|      if (warp < \{consumers}) {
    $|        #pragma unroll
    $|        for (int row = 0; row < \{fold_rows}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|          #pragma unroll
    $|          for (int item = 0; item < 8; ++item) {
    $|            const int output_row = row_begin + (consumer_row + row) * 16 + lane / 4 + (item % 4 / 2) * 8;
    $|            const int output_column = lane % 4 * 2 + item % 2 + item / 4 * 8;
    $|            if (output_row < token_count) {
    $|              const float gate = cg[row][item], up = cu[row][item];
    $|              workspace[(unsigned long long)output_row * \{intermediate}ULL + column_begin + consumer_column * 16 + output_column] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
    $|            }
    $|          }
    $|          __syncwarp();
    $|        }
    $|      }
    $|    }
    $|    return;
    $|  }
    $|
  source
}

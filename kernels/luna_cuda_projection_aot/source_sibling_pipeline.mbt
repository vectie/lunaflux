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
  let layout : CudaProjectionTransportLayout = { columns: width, }
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
          lanes=groups * 32,
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
        lanes=groups * 32,
      ),
    )
    if guard_rows {
      copies.write_string("\n}\n")
    }
    copies.write_string("\n")
  }
  let fragment_fold = projection_fragment_pipeline_source(
    fold_rows,
    width,
    rows * width,
    plan.fragment_stage_count(),
    products=[(rows * width, "cg"), ((rows + columns) * width, "cu")],
    row_limit="token_count",
  )
  let source =
    $|  if (token_count >= \{plan.minimum_rows()}) {
    $|    __shared__ __align__(32) __nv_bfloat16 sibling_storage[\{slot * plan.stage_count()}];
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
    $|        \{if projection_operand_copy_is_async(width) { "asm volatile(\"cp.async.commit_group;\" ::: \"memory\");" } else { "" }}
    $|      };
    $|      float cg[\{fold_rows}][8] = {}, cu[\{fold_rows}][8] = {};
    $|\{projection_fold_lifetime_source(plan, "if (warp < " + consumers.to_string() + ") {\nconst __nv_bfloat16 *current = sibling_storage + stage * " + slot.to_string() + ";\n" + fragment_fold + "\n        }", asynchronous=true, phase_argument=true, completed_on_issue=!projection_operand_copy_is_async(width))}
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
    $|        }
    $|      }
    $|    }
    $|    return;
    $|  }
    $|
  source
}

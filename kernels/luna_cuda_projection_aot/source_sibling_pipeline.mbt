///|
/// A product of ordered folds shares both the staged input and each pair of
/// weight tiles. Copy scheduling and CUDA lane ownership are lowering details.
fn projection_sibling_fold_pipeline_source(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
  parameters : ProjectionParameters,
  effects : @luna_projection_tile_compiler.ProjectionMaterializationEffects,
  epilogue? : @luna_projection_tile_compiler.ProjectionSiblingEpiloguePlan,
) -> String {
  if physical.sibling_reuse {
    return projection_sibling_reuse_source(
      physical,
      parameters,
      effects,
      epilogue?,
    )
  }
  let plan = physical.schedule
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let input_width = parameters.input_width
  let intermediate = parameters.intermediate_width
  let distribution = physical.distribution
  let fold_rows = distribution.rows_per_group()
  let consumers = distribution.consumer_groups()
  let operands = physical.operands
  let body = projection_packed_fold_body(
    physical,
    storage="sibling_storage",
    consumers~,
    products=[(1, "cg"), (2, "cu")],
    row_limit="token_count",
  )
  let source =
    $|  if (token_count >= \{plan.minimum_rows()}) {
    $|    __shared__ __align__(32) __nv_bfloat16 sibling_storage[\{operands.allocation_elements()}];
    $|    \{projection_operand_address_source(operands.layout())}
    $|    const int groups = \{intermediate / columns};
    $|    const int work_count = ((token_count + \{rows - 1}) / \{rows}) * groups;
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / groups) * \{rows};
    $|      const int column_begin = (work % groups) * \{columns};
    $|      const int consumer_row = (warp / \{distribution.output_groups()}) * \{fold_rows};
    $|      const int consumer_column = warp % \{distribution.output_groups()};
    $|      \{projection_sibling_transport_source(physical, input_width)}
    $|      float cg[\{fold_rows}][8] = {}, cu[\{fold_rows}][8] = {};
    $|\{projection_fold_lifetime_source(physical, body, phase_argument=physical.sibling_transport == IndependentSiblingTransport)}
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

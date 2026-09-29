///|
/// Lower the independent sibling axis to consumers, sharing each weight
/// fragment across its row product. Once the ordered fold finishes, its input
/// staging lifetime ends; reuse that storage for the gate epilogue. The up
/// epilogue uses the already admitted dynamic scratch. CTA fences delimit both
/// changes of ownership, including inactive tail-row consumers.
/// CUDA lowering uses the documented m16n8k16 register layout, not the opaque
/// WMMA fragment representation. Two independent column halves retain the same
/// ordered K16 fold. Row-contiguous swizzled operands feed ldmatrix; register-major
/// result planes make each warp's shared stores and subsequent loads contiguous.
fn projection_sibling_reuse_source(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
  parameters : ProjectionParameters,
  effects : @luna_projection_tile_compiler.ProjectionMaterializationEffects,
) -> String {
  let plan = physical.schedule
  let reuse = physical.distribution
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let operands = physical.operands
  let (gate_offset, _, _) = operands.region(1).unwrap().geometry()
  let (up_offset, _, _) = operands.region(2).unwrap().geometry()
  let folds = reuse.rows_per_group()
  let consumers = reuse.consumer_groups()
  let input_width = parameters.input_width
  let intermediate = parameters.intermediate_width
  let body = projection_packed_fold_body(
    physical,
    storage="sibling_storage",
    consumers=consumers * 2,
    row_limit="token_count",
    weight_displacement="sibling * \{up_offset - gate_offset}",
  )
  let source =
    $|  if (token_count >= \{plan.minimum_rows()}) {
    $|    __shared__ __align__(32) __nv_bfloat16 sibling_storage[\{Int::max(operands.allocation_elements(), rows * columns * 2)}];
    $|    extern __shared__ unsigned char sibling_output_bytes[];
    $|    const int groups = \{intermediate / columns};
    $|    const int work_count = ((token_count + \{rows - 1}) / \{rows}) * groups;
    $|    const int sibling = warp / \{consumers};
    $|    const int consumer = warp % \{consumers};
    $|    const int consumer_row = (consumer / \{reuse.output_groups()}) * \{folds};
    $|    const int consumer_column = consumer % \{reuse.output_groups()};
    $|    \{projection_operand_address_source(operands.layout())}
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / groups) * \{rows};
    $|      const int column_begin = (work % groups) * \{columns};
    $|      \{projection_sibling_phased_transfer_source(physical.operands, physical.transfer_segments, input_width)}
    $|      float c[\{folds}][8] = {};
    $|\{projection_fold_lifetime_source(physical, body)}
    $|      float *gate_tiles = reinterpret_cast<float *>(sibling_storage);
    $|      float *up_tiles = reinterpret_cast<float *>(sibling_output_bytes);
    $|      if (warp < \{consumers * 2}) {
    $|        float *result_tiles = sibling == 0 ? gate_tiles : up_tiles;
    $|        #pragma unroll
    $|        for (int row = 0; row < \{folds}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|          #pragma unroll
    $|          for (int value = 0; value < 8; ++value)
    $|            result_tiles[((consumer_row + row) * \{reuse.output_groups()} + consumer_column) * 256 + value * 32 + lane] = c[row][value];
    $|        }
    $|      }
    $|      \{projection_materialization_fence_source(Some(effects.publication()))}
    $|      for (int element = threadIdx.x; element < \{rows * columns}; element += blockDim.x) {
    $|        const int tile = element / 256, cell = element % 256;
    $|        const int value = cell / 32, owner_lane = cell % 32;
    $|        const int output_row = row_begin + (tile / \{reuse.output_groups()}) * 16 + owner_lane / 4 + ((value % 4) / 2) * 8;
    $|        if (output_row < token_count) {
    $|          const float gate = gate_tiles[element], up = up_tiles[element];
    $|          workspace[(unsigned long long)output_row * \{intermediate}ULL + column_begin + (tile % \{reuse.output_groups()}) * 16 + (owner_lane % 4) * 2 + value % 2 + (value / 4) * 8] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
    $|        }
    $|      }
    $|      \{projection_materialization_fence_source(effects.release())}
    $|    }
    $|    return;
    $|  }
    $|
  source
}

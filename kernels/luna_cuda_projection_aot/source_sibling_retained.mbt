///|
/// CUDA realization of producer-retained epilogue ownership. Matching gate/up
/// fragments have identical register-major coordinates. Only up is published;
/// gate owners keep their accumulators and produce the final BF16 directly.
/// The operand ring's final uniform retirement precedes its alias as one peer
/// plane, and the final reader retirement precedes the next work item.
fn projection_retained_sibling_epilogue_source(
  plan : @luna_projection_tile_compiler.ProjectionSiblingEpiloguePlan,
  parameters : ProjectionParameters,
) -> String {
  let reuse = plan.distribution()
  let folds = reuse.rows_per_group()
  let consumers = reuse.consumer_groups()
  let outputs = reuse.output_groups()
  let effects = plan.effects().unwrap()
  guard plan.result_planes() == 1 else {
    abort("retained sibling epilogue requires exactly one peer plane")
  }
  (
    $|      float *peer_tiles = reinterpret_cast<float *>(sibling_storage);
    $|      if (warp >= \{consumers} && warp < \{consumers * 2}) {
    $|        #pragma unroll
    $|        for (int row = 0; row < \{folds}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|          #pragma unroll
    $|          for (int value = 0; value < 8; ++value)
    $|            peer_tiles[((consumer_row + row) * \{outputs} + consumer_column) * 256 + value * 32 + lane] = c[row][value];
    $|        }
    $|      }
    $|      \{projection_materialization_fence_source(Some(effects.publication()))}
    $|      if (warp < \{consumers}) {
    $|        #pragma unroll
    $|        for (int row = 0; row < \{folds}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|          #pragma unroll
    $|          for (int value = 0; value < 8; ++value) {
    $|            const int output_row = row_begin + (consumer_row + row) * 16 + lane / 4 + ((value % 4) / 2) * 8;
    $|            if (output_row < token_count) {
    $|              const float gate = c[row][value];
    $|              const float up = peer_tiles[((consumer_row + row) * \{outputs} + consumer_column) * 256 + value * 32 + lane];
    $|              workspace[(unsigned long long)output_row * \{parameters.intermediate_width}ULL + column_begin + consumer_column * 16 + (lane % 4) * 2 + value % 2 + (value / 4) * 8] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
    $|            }
    $|          }
    $|        }
    $|      }
    $|      \{projection_materialization_fence_source(effects.release())}
  )
}

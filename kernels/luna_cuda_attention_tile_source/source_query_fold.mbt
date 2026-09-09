///|
/// Realize the portable finite product as named scalar state, not an array
/// indexed by runtime query/warp_count. This makes address-taking impossible
/// for the carried state without relying on a backend loop-unroll heuristic.
fn cuda_query_fold_declaration(offsets : ReadOnlyArray[Int]) -> String {
  offsets
  .iter()
  .map(offset => {
    "  float fold_maximum_\{offset} = -CUDART_INF_F;\n" +
    "  float fold_denominator_\{offset} = 0.0f;\n"
  })
  .to_array()
  .join("")
}

///|
fn cuda_query_fold_update(
  offsets : ReadOnlyArray[Int],
  probability_store : String,
) -> String {
  offsets
  .iter()
  .map(offset => {
    (
      $|    {
      $|      const int query = warp + \{offset};
      $|      if (query < active_queries) {
      $|        const int query_position = query_positions[token_base + query];
      $|        float local_maximum = -CUDART_INF_F;
      $|        for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32)
      $|          local_maximum = fmaxf(local_maximum, lf_transformed_score(scores, query, key, key_base, partition_end, query_position, scale));
      $|        const float tile_maximum = lf_warp_max(local_maximum);
      $|        const float previous_maximum = fold_maximum_\{offset};
      $|        const float next_maximum = fmaxf(previous_maximum, tile_maximum);
      $|        const float previous_scale = isinf(previous_maximum)
      $|            ? 0.0f : expf(previous_maximum - next_maximum);
      $|        float local_sum = 0.0f;
      $|        for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32) {
      $|          const float weight = isinf(next_maximum) ? 0.0f :
      $|              expf(lf_transformed_score(scores, query, key, key_base, partition_end, query_position, scale) - next_maximum);
      $|\{probability_store}
      $|          local_sum += weight;
      $|        }
      $|        const float tile_sum = lf_warp_sum(local_sum);
      $|        fold_maximum_\{offset} = next_maximum;
      $|        fold_denominator_\{offset} =
      $|            fold_denominator_\{offset} * previous_scale + tile_sum;
      $|        if (lane == 0) previous_scales[query] = previous_scale;
      $|      }
      $|    }
      $|
    )
  })
  .to_array()
  .join("") +
  "    __syncthreads();\n"
}

///|
fn cuda_query_fold_store(offsets : ReadOnlyArray[Int]) -> String {
  offsets
  .iter()
  .map(offset => {
    (
      $|  if (warp + \{offset} < active_queries && lane == 0) {
      $|    running_maximum[warp + \{offset}] = fold_maximum_\{offset};
      $|    running_denominator[warp + \{offset}] = fold_denominator_\{offset};
      $|  }
      $|
    )
  })
  .to_array()
  .join("") +
  "  __syncthreads();\n"
}

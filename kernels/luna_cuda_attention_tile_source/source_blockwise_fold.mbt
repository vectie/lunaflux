///|
/// Interpret the pure blockwise state program. Shared score ownership is
/// disjoint per subgroup; its exchanges do not require a workgroup barrier.
fn cuda_blockwise_fold_body(
  fold : @attention_physical_ir.BlockwiseFold,
  value_ready? : String = "",
) -> String {
  fold
  .steps()
  .map(step => {
    match step {
      ComputeTileScores =>
        (
          #|    for (int key = 0; key < active_keys; ++key) {
          #|      float dot = 0.0f;
          #|      #pragma unroll
          #|      for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {
          #|        const int component = lane + item * 32;
          #|        if (component < LF_HEAD_DIMENSION)
          #|          dot += query_fragment[item] * __bfloat162float(key_tile[key * LF_HEAD_DIMENSION + component]);
          #|      }
          #|      for (int offset = 16; offset > 0; offset >>= 1)
          #|        dot += __shfl_down_sync(0xffffffffU, dot, offset);
          #|      if (lane == 0) scores[key] = dot * scale;
          #|    }
        )
      PublishTileScores => "    __syncwarp();"
      ReduceTileMaximum =>
        (
          #|    float tile_maximum = -CUDART_INF_F;
          #|    for (int key = lane; key < active_keys; key += 32)
          #|      tile_maximum = fmaxf(tile_maximum, scores[key]);
          #|    for (int offset = 16; offset > 0; offset >>= 1)
          #|      tile_maximum = fmaxf(tile_maximum, __shfl_down_sync(0xffffffffU, tile_maximum, offset));
          #|    tile_maximum = __shfl_sync(0xffffffffU, tile_maximum, 0);
        )
      FormFloat32Probabilities =>
        (
          #|    float tile_denominator = 0.0f;
          #|    for (int key = lane; key < active_keys; key += 32) {
          #|      const float probability = expf(scores[key] - tile_maximum);
          #|      scores[key] = probability;
          #|      tile_denominator += probability;
          #|    }
        )
      ReduceTileDenominator =>
        (
          #|    for (int offset = 16; offset > 0; offset >>= 1)
          #|      tile_denominator += __shfl_down_sync(0xffffffffU, tile_denominator, offset);
          #|    tile_denominator = __shfl_sync(0xffffffffU, tile_denominator, 0);
        )
      PublishTileProbabilities => "    __syncwarp();"
      AccumulateTileValues =>
        value_ready +
        "\n" +
        (
          #|    float tile_numerator[LF_ACCUMULATOR_ITEMS] = {};
          #|    for (int key = 0; key < active_keys; ++key) {
          #|      const float probability = scores[key];
          #|      #pragma unroll
          #|      for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {
          #|        const int component = lane + item * 32;
          #|        if (component < LF_HEAD_DIMENSION)
          #|          tile_numerator[item] += probability * __bfloat162float(value_tile[key * LF_HEAD_DIMENSION + component]);
          #|      }
          #|    }
        )
      MergeTileState =>
        (
          #|    const float next_maximum = fmaxf(maximum, tile_maximum);
          #|    const float old_scale = denominator == 0.0f ? 0.0f : expf(maximum - next_maximum);
          #|    const float tile_scale = expf(tile_maximum - next_maximum);
          #|    denominator = denominator * old_scale + tile_denominator * tile_scale;
          #|    maximum = next_maximum;
          #|    #pragma unroll
          #|    for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item)
          #|      accumulator[item] = accumulator[item] * old_scale + tile_numerator[item] * tile_scale;
        )
    }
  })
  .join("\n")
}

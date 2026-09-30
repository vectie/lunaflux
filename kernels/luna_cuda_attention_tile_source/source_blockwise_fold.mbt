///|
/// Interpret the pure blockwise state program. Shared score ownership is
/// disjoint per subgroup; its exchanges do not require a workgroup barrier.
fn cuda_blockwise_fold_body(
  fold : @attention_physical_ir.BlockwiseFold,
  value_ready? : String = "",
) -> String {
  let dimensions = fold.dimensions().0
  let ownership = @attention_physical_ir.ValueComponentOwnership::new(
    dimensions~,
    owners=32,
    maximum_fragment=if dimensions == 128 { 4 } else { 1 },
  ).unwrap()
  let accumulate = if ownership.fragment_width() == 4 {
    (
      #|    float tile_numerator[LF_ACCUMULATOR_ITEMS] = {};
      #|    for (int key = 0; key < active_keys; ++key) {
      #|      const float probability = scores[key];
      #|      const uint2 packed = *reinterpret_cast<const uint2 *>(value_tile + key * LF_HEAD_DIMENSION + lane * 4);
      #|      const float values[4] = { __uint_as_float(packed.x << 16), __uint_as_float(packed.x & 0xffff0000U),
      #|                                __uint_as_float(packed.y << 16), __uint_as_float(packed.y & 0xffff0000U) };
      #|      #pragma unroll
      #|      for (int item = 0; item < 4; ++item)
      #|        tile_numerator[item] += probability * values[item];
      #|    }
    )
  } else {
    (
      #|    float tile_numerator[LF_ACCUMULATOR_ITEMS] = {};
      #|    for (int key = 0; key < active_keys; ++key) {
      #|      const float probability = scores[key];
      #|      #pragma unroll
      #|      for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {
      #|        const int component = LF_VALUE_COMPONENT(item);
      #|        if (component < LF_HEAD_DIMENSION)
      #|          tile_numerator[item] += probability * __bfloat162float(value_tile[key * LF_HEAD_DIMENSION + component]);
      #|      }
      #|    }
    )
  }
  let ordered = fold
    .steps()
    .map(step => {
      match step {
        ComputeTileScores =>
          (
            #|    for (int key_group = 0; key_group < active_keys; key_group += LF_SCORE_OWNERS) {
            #|      const int key = key_group + score_owner;
            #|      float dot = 0.0f;
            #|      #pragma unroll
            #|      for (int item = 0; item < LF_SCORE_QUERY_ITEMS; ++item) {
            #|        const int component = LF_SCORE_COMPONENT(item);
            #|        if (key < active_keys && component < LF_HEAD_DIMENSION)
            #|          dot += query_fragment[item] * __bfloat162float(key_tile[key * LF_HEAD_DIMENSION + component]);
            #|      }
            #|      for (int offset = LF_SCORE_LANES / 2; offset > 0; offset >>= 1)
            #|        dot += __shfl_down_sync(0xffffffffU, dot, offset, LF_SCORE_LANES);
            #|      if (score_lane == 0 && key < active_keys) scores[key] = dot * scale;
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
        AccumulateTileValues => value_ready + "\n" + accumulate
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
  if !fold.contracts_multiply_add() {
    ordered
  } else {
    // Contraction is limited to the operations declared by the physical fold.
    // The module still compiles with global --fmad=false; expf, owner reduction
    // order and probability precision are unchanged.
    ordered
    .replace(
      old="dot += query_fragment[item] * __bfloat162float(key_tile[key * LF_HEAD_DIMENSION + component]);",
      new="dot = fmaf(query_fragment[item], __bfloat162float(key_tile[key * LF_HEAD_DIMENSION + component]), dot);",
    )
    .replace(
      old="tile_numerator[item] += probability * values[item];",
      new="tile_numerator[item] = fmaf(probability, values[item], tile_numerator[item]);",
    )
    .replace(
      old="tile_numerator[item] += probability * __bfloat162float(value_tile[key * LF_HEAD_DIMENSION + component]);",
      new="tile_numerator[item] = fmaf(probability, __bfloat162float(value_tile[key * LF_HEAD_DIMENSION + component]), tile_numerator[item]);",
    )
    .replace(
      old="denominator = denominator * old_scale + tile_denominator * tile_scale;",
      new="denominator = fmaf(denominator, old_scale, tile_denominator * tile_scale);",
    )
    .replace(
      old="accumulator[item] = accumulator[item] * old_scale + tile_numerator[item] * tile_scale;",
      new="accumulator[item] = fmaf(accumulator[item], old_scale, tile_numerator[item] * tile_scale);",
    )
  }
}

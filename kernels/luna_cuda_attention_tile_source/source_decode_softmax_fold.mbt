///|
/// Materialize the finite score map before consuming its value tile. The
/// named product has no addressable score array or dynamic index. Register
/// allocation itself remains the offline backend compiler's responsibility.
fn emit_decode_softmax_score_map(
  dot : String,
  offsets : ReadOnlyArray[Int],
) -> String {
  "    float tile_maximum = -CUDART_INF_F;\n" +
  offsets
  .iter()
  .map(offset => {
    (
      $|    float tile_score_\{offset} = 0.0f;
      $|    {
      $|      const int key = split + \{offset};
      $|      if (key < active_keys) {
      $|\{dot}
      $|        tile_score_\{offset} = score;
      $|        if (lane == 0) tile_maximum = fmaxf(tile_maximum, score);
      $|      }
      $|    }
      $|
    )
  })
  .to_array()
  .join("")
}

///|
/// Realize associative tile-softmax folding without changing any shared-memory
/// producer/consumer layout. Rescale the running vector once per tile, not once
/// per key. Lane zero owns scalar state; one weight broadcast serves every
/// head component. An empty strided partition preserves the merge identity.
/// This changes F32 evaluation order and requires tolerant numerical checks.
fn emit_decode_softmax_tile_fold(
  accumulator_items : Int,
  offsets : ReadOnlyArray[Int],
) -> String {
  let rescale =
    $|    // Associative tile-softmax: one running-state rescale per key tile.
    $|    float tile_old_scale = 1.0f;
    $|    if (lane == 0 && split < active_keys) {
    $|      const float next_maximum = fmaxf(maximum, tile_maximum);
    $|      tile_old_scale = denominator == 0.0f
    $|          ? 0.0f : expf(maximum - next_maximum);
    $|      maximum = next_maximum;
    $|      denominator *= tile_old_scale;
    $|    }
    $|    tile_old_scale = __shfl_sync(0xffffffffU, tile_old_scale, 0);
    $|    #pragma unroll
    $|    for (int item = 0; item < \{accumulator_items}; ++item)
    $|      accumulator[item] *= tile_old_scale;
    $|
  rescale +
  offsets
  .iter()
  .map(offset => {
    (
      $|    {
      $|      const int key = split + \{offset};
      $|      if (key < active_keys) {
      $|        float weight = 0.0f;
      $|        if (lane == 0) {
      $|          weight = expf(tile_score_\{offset} - maximum);
      $|          denominator += weight;
      $|        }
      $|        weight = __shfl_sync(0xffffffffU, weight, 0);
      $|        #pragma unroll
      $|        for (int item = 0; item < \{accumulator_items}; ++item) {
      $|          const int component = lane + item * 32;
      $|          if (component < LF_HEAD_DIMENSION)
      $|            accumulator[item] += weight * __bfloat162float(
      $|                value_tile[key * LF_HEAD_DIMENSION + component]);
      $|        }
      $|      }
      $|    }
      $|
    )
  })
  .to_array()
  .join("")
}

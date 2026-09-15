///|
fn cuda_score_bootstrap(
  point : @luna_attention_tile_schedule.AttentionScoreBootstrapPoint,
  resident_output : Bool,
  initial_accumulator : String,
  input_row_width : Int,
  shared_statistics : String,
) -> String {
  let actions = @luna_attention_tile_schedule.plan_attention_score_bootstrap(
    point,
    resident_output~,
  )
  let result = actions
    .map(action => {
      match action {
        InitializeScoreOutput =>
          "  for (int linear = thread; linear < LF_QUERY_TILE * LF_HEAD_DIMENSION; linear += LF_BLOCK_THREADS)\n" +
          "    accumulator[lf_float_index(linear / LF_HEAD_DIMENSION, linear % LF_HEAD_DIMENSION, LF_HEAD_DIMENSION)] = \{initial_accumulator};\n"
        PublishBootstrapWrites => "  __syncthreads();\n"
        InitializeResidentScoreOutput => cuda_resident_output_fold_declaration()
        InitializeScoreStatistics => shared_statistics + "\n"
        StageScoreQueries =>
          (
            $|  const int query_vector_count = LF_QUERY_TILE * LF_HEAD_DIMENSION / LF_VALUES_PER_VECTOR;
            $|  for (int vector = thread; vector < query_vector_count; vector += LF_BLOCK_THREADS) {
            $|    const int first_value = vector * LF_VALUES_PER_VECTOR;
            $|    const int query = first_value / LF_HEAD_DIMENSION;
            $|    const int component = first_value % LF_HEAD_DIMENSION;
            $|    uint4 packed = make_uint4(0U, 0U, 0U, 0U);
            $|    if (query < active_queries) {
            $|      const unsigned long long source_index =
            $|          (unsigned long long)(token_base + query) * \{input_row_width}ULL +
            $|          (unsigned long long)query_head * LF_HEAD_DIMENSION + component;
            $|      packed = *reinterpret_cast<const uint4 *>(rotated_qkv_rows + source_index);
            $|    }
            $|    *reinterpret_cast<uint4 *>(query_tile + lf_operand_index(query, component, LF_HEAD_DIMENSION)) = packed;
            $|  }
            $|
          )
        ValidateScoreQueryBounds =>
          (
            #|  int maximum_query_position = query_positions[token_base];
            #|  if (thread == 0) {
            #|    tile_valid = 1;
            #|    for (int query = 0; query < active_queries; ++query) {
            #|      const int position = query_positions[token_base + query];
            #|      if (position < 0 || position >= sequence_length) tile_valid = 0;
            #|      maximum_query_position = max(maximum_query_position, position);
            #|    }
            #|    scores[0] = __int_as_float(maximum_query_position);
            #|  }
            #|
          )
        ReadScoreQueryBounds =>
          "  maximum_query_position = __float_as_int(scores[0]);\n"
        RequireValidScoreQueries => "  if (!tile_valid) return;\n"
        ReleaseScoreValidationReaders =>
          "  // Complete query-bounds/validation reads before the next staging epoch.\n  __syncthreads();\n"
      }
    })
    .join("")
  // Preserve the surrounding source template's existing line separation.
  if point == InitializeScoreAccumulators {
    result
  } else {
    result.trim_end().to_owned()
  }
}

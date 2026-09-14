///|
/// Block-owned QK/PV schedules supported by the current lowering. Geometry is
/// generated independently of compatibility IDs. Async overlap currently has
/// a narrower realized domain than synchronous transfer.
fn block_owned_matrix_specs() -> Array[AttentionCandidateSpec] {
  let specs : Array[AttentionCandidateSpec] = []
  for query_index, query in [16, 32, 64] {
    for key_index, keys in [32, 64] {
      specs.push({
        stable_id: 310 + query_index * 2 + key_index,
        phase: Prefill,
        compute: AttentionMatrixQkPv,
        memory: AttentionStagedShared,
        query_rows: query,
        key_value_rows: keys,
        subgroups_per_block: 8,
        pipeline_stages: 1,
        vector_bytes: 16,
        minimum_query_rows: query,
        // Preserve the historical small-query profitability threshold.
        minimum_context_tokens: if query == 16 {
          32
        } else {
          keys
        },
      })
    }
  }
  for index, query in [32, 64] {
    specs.push({
      stable_id: 316 + index,
      phase: Prefill,
      compute: AttentionMatrixQkPv,
      memory: AttentionAsyncStagedShared,
      query_rows: query,
      key_value_rows: 32,
      subgroups_per_block: 8,
      pipeline_stages: 2,
      vector_bytes: 16,
      minimum_query_rows: query,
      minimum_context_tokens: 64,
    })
  }
  specs
}

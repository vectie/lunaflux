///|
/// Compatibility names are serialization identities, not selection rules.
/// New Cartesian points use a disjoint, injective identity range.
fn query_owned_candidate_id(query : Int, keys : Int, pipeline : Int) -> Int {
  match (query, keys, pipeline) {
    (64, 64, 0) => 318
    (32, 32, 0) => 319
    (64, 64, 2) => 320
    (32, 32, 2) => 321
    (64, 64, 1) => 322
    (32, 32, 1) => 323
    (64, 32, 1) => 324
    _ => 1000 + (query / 32 - 1) * 6 + (keys / 32 - 1) * 3 + pipeline
  }
}

///|
/// Enumerate the complete currently supported ownership domain, rather than
/// enumerating hand-picked schedules. Resource and numeric constraints are
/// applied by make_attention_candidate, after this pure Cartesian expansion.
/// 32/64 are the current ownership implementation domain, not latency weights;
/// expanding that domain requires corresponding backend realizations.
fn query_owned_attention_specs() -> Array[AttentionCandidateSpec] {
  let specs : Array[AttentionCandidateSpec] = []
  for query in [32, 64] {
    for keys in [32, 64] {
      for pipeline in [0, 1, 2] {
        specs.push({
          stable_id: query_owned_candidate_id(query, keys, pipeline),
          phase: Prefill,
          compute: AttentionMatrixQkPv,
          memory: if pipeline == 0 {
            AttentionStagedShared
          } else {
            AttentionAsyncStagedShared
          },
          query_rows: query,
          key_value_rows: keys,
          subgroups_per_block: query / 16,
          pipeline_stages: if pipeline == 0 {
            1
          } else {
            pipeline
          },
          vector_bytes: 16,
          minimum_query_rows: query,
          minimum_context_tokens: keys,
        })
      }
    }
  }
  specs
}

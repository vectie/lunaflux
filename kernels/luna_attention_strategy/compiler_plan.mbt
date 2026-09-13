///|
fn validate_attention_autotune_records(
  records : ArrayView[AttentionAutotuneRecord],
) -> Unit raise AttentionStrategyError {
  for left_index, left in records {
    let mut right_index = left_index + 1
    while right_index < records.length() {
      let right = records[right_index]
      if left.key == right.key {
        if left.median_latency_ns == right.median_latency_ns &&
          left.sample_count == right.sample_count {
          raise InvalidAttentionStrategy(Duplicate)
        }
        raise InvalidAttentionStrategy(Conflict)
      }
      let left_key = left.key
      let right_key = right.key
      if left_key.backend == right_key.backend &&
        left_key.target == right_key.target &&
        left_key.problem == right_key.problem &&
        left_key.candidate.stable_id == right_key.candidate.stable_id {
        raise InvalidAttentionStrategy(Conflict)
      }
      right_index += 1
    }
  }
}

///|
/// Compiles one serving bucket. Exact matching offline measurements override
/// the deterministic cost model; ties are resolved by stable candidate ID.
pub fn compile_attention_tile_plan(
  records : ArrayView[AttentionAutotuneRecord],
  backend~ : AttentionBackendId,
  target~ : AttentionTargetId,
  problem~ : AttentionCompileProblem,
  capabilities~ : AttentionCompilerCapabilities,
) -> CompiledAttentionTilePlan raise AttentionStrategyError {
  validate_attention_autotune_records(records)
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  let initial = candidates[0]
  let (selected, selected_cost) = candidates[1:].fold(
    init=(initial, attention_candidate_static_cost(problem, initial)),
    (current, candidate) => {
      let (current_candidate, current_cost) = current
      let cost = attention_candidate_static_cost(problem, candidate)
      if cost < current_cost ||
        (
          cost == current_cost &&
          candidate.stable_id < current_candidate.stable_id
        ) {
        (candidate, cost)
      } else {
        current
      }
    },
  )
  let selected_record = records.fold(init=None, (
    current : AttentionAutotuneRecord?,
    record,
  ) => {
    let key = record.key
    if key.backend == backend &&
      key.target == target &&
      key.problem == problem &&
      attention_candidate_compatible(problem, capabilities, key.candidate) {
      let candidate_present = candidates.any(candidate => {
        candidate == key.candidate
      })
      if candidate_present {
        match current {
          None => Some(record)
          Some(selected) =>
            if record.median_latency_ns < selected.median_latency_ns ||
              (
                record.median_latency_ns == selected.median_latency_ns &&
                record.key.candidate.stable_id <
                selected.key.candidate.stable_id
              ) {
              Some(record)
            } else {
              current
            }
        }
      } else {
        current
      }
    } else {
      current
    }
  })
  match selected_record {
    Some(record) =>
      {
        problem,
        candidate: record.key.candidate,
        source: AttentionOfflineAutotune,
        static_cost: attention_candidate_static_cost(
          problem,
          record.key.candidate,
        ),
      }
    None =>
      {
        problem,
        candidate: selected,
        source: AttentionStaticCostModel,
        static_cost: selected_cost,
      }
  }
}

///|
/// Reifies one generated candidate as a static compiler plan. The functional
/// compiler uses this entry point to feed post-rewrite resource information
/// back into strategy selection without constructing or mutating strategy
/// records. Exact membership in the bounded candidate set is required.
pub fn compile_attention_tile_candidate_plan(
  problem : AttentionCompileProblem,
  capabilities : AttentionCompilerCapabilities,
  candidate : AttentionTileCandidate,
) -> CompiledAttentionTilePlan raise AttentionStrategyError {
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  if !candidates.any(generated => generated == candidate) {
    raise InvalidAttentionStrategy(Candidate)
  }
  {
    problem,
    candidate,
    source: AttentionStaticCostModel,
    static_cost: attention_candidate_static_cost(problem, candidate),
  }
}

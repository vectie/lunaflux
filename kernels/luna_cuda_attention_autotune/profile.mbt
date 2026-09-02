///|
pub fn cuda_sm120_rtx5060ti_backend() -> @luna_attention_strategy.AttentionBackendId raise {
  @luna_attention_strategy.AttentionBackendId::new("cuda")
}

///|
pub fn cuda_sm120_rtx5060ti_target() -> @luna_attention_strategy.AttentionTargetId raise {
  @luna_attention_strategy.AttentionTargetId::new("sm120-rtx5060ti-bf16-v1")
}

///|
fn exact_candidate(
  candidates : ReadOnlyArray[@luna_attention_strategy.AttentionTileCandidate],
  stable_id : Int,
) -> @luna_attention_strategy.AttentionTileCandidate raise {
  guard candidates.filter(candidate => candidate.stable_id() == stable_id)
    is [candidate] else {
    raise Failure("CUDA attention autotune candidate is unavailable")
  }
  candidate
}

///|
/// Immutable replay of the sm120 RTX 5060 Ti BF16 prefill measurement made
/// over the 16, 65, 128, and ragged 17+65 token vectors. Runtime compilation
/// performs no benchmarking or mutable cache lookup.
pub fn cuda_sm120_rtx5060ti_prefill_records(
  problem : @luna_attention_strategy.AttentionCompileProblem,
  capabilities : @luna_attention_strategy.AttentionCompilerCapabilities,
) -> ReadOnlyArray[@luna_attention_strategy.AttentionAutotuneRecord] raise {
  if problem.phase() != Prefill ||
    problem.numeric_type() != AttentionBFloat16 ||
    problem.dimensions() != (128, 4096) ||
    problem.shape().dimensions() != (16, 4, 128, 16) ||
    !problem.is_causal() {
    return ReadOnlyArray::from_array([])
  }
  let backend = cuda_sm120_rtx5060ti_backend()
  let target = cuda_sm120_rtx5060ti_target()
  let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
    problem, capabilities,
  )
  let measured : Array[(Int, Int64)] = [
    (290, 56506L),
    (291, 106508L),
    (300, 67645L),
    (310, 31048L),
    (311, 47146L),
  ]
  ReadOnlyArray::from_array(
    measured.map(item => {
      let (stable_id, median_latency_ns) = item
      let candidate = exact_candidate(candidates, stable_id)
      @luna_attention_strategy.AttentionAutotuneRecord::new(
        key=@luna_attention_strategy.AttentionAutotuneKey::new(
          backend~,
          target~,
          problem~,
          candidate~,
        ),
        median_latency_ns~,
        sample_count=100,
      )
    }),
  )
}

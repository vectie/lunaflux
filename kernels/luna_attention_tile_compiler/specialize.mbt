///|
/// Offline specialization of an explicitly chosen legal alternative. This is
/// not an autotune observation: no invented timing is needed to compile a
/// candidate for measurement. The ordinary semantic, effects and resource
/// passes remain mandatory and the returned schedule retains its identity.
pub fn specialize_partitioned_attention_tile(
  request : AttentionTileCompilerRequest,
  candidate_id : Int,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  let plans = @luna_attention_strategy.compile_attention_tile_candidate_plans(
    request.problem,
    request.capabilities,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(StrategySelection)
  }
  guard plans.filter(p => p.candidate().stable_id() == candidate_id) is [plan] else {
    raise InvalidAttentionTileCompilation(StrategySelection)
  }
  let compilation = schedule_partitioned_attention_tile(
    optimize_attention_tile(elaborate_attention_tile({ request, plan, })),
  )
  if attention_tile_resource_score(request, compilation) == 9223372036854775807L {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  compilation
}

///|
/// Explicit offline evaluation is not an autotune observation. Retain that
/// distinction in the immutable frontier identity and recheck its resources.
pub fn AttentionTileCompilationFrontier::specialize(
  self : AttentionTileCompilationFrontier,
  request : AttentionTileCompilerRequest,
  candidate_id : Int,
) -> AttentionTileCompilationFrontier raise AttentionTileCompilerError {
  guard self.variants.filter(v => {
      v.plan().candidate().stable_id() == candidate_id
    })
    is [selected] else {
    raise InvalidAttentionTileCompilation(StrategySelection)
  }
  if attention_tile_resource_score(request, selected) == 9223372036854775807L {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let canonical = @utf8.encode(
    "luna-attention-explicit-specialization-v1\nbase=" +
    self.digest.as_hex() +
    "\nselected=" +
    selected.digest().as_hex() +
    "\n",
  )
  {
    ..self,
    selected,
    canonical,
    digest: AttentionTileCompilationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}

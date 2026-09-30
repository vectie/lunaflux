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

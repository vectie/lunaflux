///|
/// Offline specialization of an explicitly chosen legal alternative. This is
/// not an autotune observation: no invented timing is needed to compile a
/// candidate for measurement. The ordinary semantic, effects and resource
/// passes remain mandatory and the returned schedule retains its identity.
pub fn specialize_partitioned_attention_tile(
  request : AttentionTileCompilerRequest,
  candidate_id : Int32,
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
/// Restrict an unmeasured frontier to actually implemented backend candidates,
/// then apply the same resource score and stable tie order as ordinary selection.
/// A measured incumbent cannot be replaced by a backend availability filter.
/// Supply the same request used to construct this frontier.
pub fn AttentionTileCompilationFrontier::restrict_to(
  self : Self,
  request : AttentionTileCompilerRequest,
  candidate_ids : ArrayView[Int32],
) -> Self raise AttentionTileCompilerError {
  let variants = self.variants.filter(variant => {
    candidate_ids.contains(variant.plan().candidate().stable_id())
  })
  guard !variants.is_empty() else {
    raise InvalidAttentionTileCompilation(StrategySelection)
  }
  guard self.selected.plan().source() != AttentionOfflineAutotune ||
    candidate_ids.contains(self.selected.plan().candidate().stable_id()) else {
    raise InvalidAttentionTileCompilation(StrategySelection)
  }
  let (selected, score) = variants[1:].fold(
    init=(variants[0], attention_tile_resource_score(request, variants[0])),
    (incumbent, candidate) => {
      let candidate_score = attention_tile_resource_score(request, candidate)
      if candidate_score < incumbent.1 ||
        (
          candidate_score == incumbent.1 &&
          candidate.plan().candidate().stable_id() <
          incumbent.0.plan().candidate().stable_id()
        ) {
        (candidate, candidate_score)
      } else {
        incumbent
      }
    },
  )
  guard score != 9223372036854775807L else {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let canonical = @utf8.encode(
    "luna-attention-backend-restriction-v1\nbase=" +
    self.digest.as_hex() +
    "\nselected=" +
    selected.digest().as_hex() +
    "\nvariants=" +
    variants
    .map(variant => variant.plan().candidate().stable_id().to_string())
    .join(",") +
    "\n",
  )
  {
    selected,
    variants,
    canonical,
    digest: AttentionTileCompilationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}

///|
/// Explicit offline evaluation is not an autotune observation. Retain that
/// distinction in the immutable frontier identity and recheck its resources.
pub fn AttentionTileCompilationFrontier::specialize(
  self : AttentionTileCompilationFrontier,
  request : AttentionTileCompilerRequest,
  candidate_id : Int32,
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

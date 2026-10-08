///|
/// Enumerate executable partition-map/fold/merge alternatives before choosing
/// a kernel or a parallelism target. Ordinary single-launch timing records do
/// not rank this search: they did not measure the partial-plus-merge chain.
/// Candidate IDs are an explicit, bounded search domain, not measurements.
pub fn enumerate_partitioned_attention_tiles(
  request : AttentionTileCompilerRequest,
  candidate_ids : ArrayView[Int],
  parallel_workgroup_targets : ArrayView[Int],
) -> AttentionPartitionedTileCompilationFrontier raise AttentionTileCompilerError {
  if candidate_ids.is_empty() ||
    candidate_ids.length() > 64 ||
    parallel_workgroup_targets.is_empty() ||
    parallel_workgroup_targets.length() > 16 {
    raise InvalidAttentionTileCompilation(StrategySelection)
  }
  let ids = candidate_ids.to_owned()
  ids.sort()
  for index, id in ids {
    if id <= 0 || (index > 0 && ids[index - 1] == id) {
      raise InvalidAttentionTileCompilation(StrategySelection)
    }
  }
  let targets = parallel_workgroup_targets.to_owned()
  targets.sort()
  for index, target in targets {
    if target <= 0 || (index > 0 && targets[index - 1] == target) {
      raise InvalidAttentionTileCompilation(ResourceFeedback)
    }
  }
  let variants : Array[AttentionPartitionedTileCompilationVariant] = []
  for target in targets {
    let capabilities = request.capabilities.with_parallel_workgroup_target(
      target,
    ) catch {
      _ => raise InvalidAttentionTileCompilation(ResourceFeedback)
    }
    let input = { ..request, capabilities, autotune_records: [], }
    // Enumerate once per target. No independent winner is chosen first.
    let plans = @luna_attention_strategy.compile_attention_tile_candidate_plans(
      input.problem,
      capabilities,
    ) catch {
      _ => raise InvalidAttentionTileCompilation(StrategySelection)
    }
    for id in ids {
      guard plans.filter(p => p.candidate().stable_id() == id) is [plan] else {
        raise InvalidAttentionTileCompilation(StrategySelection)
      }
      let compilation = schedule_partitioned_attention_tile(
        optimize_attention_tile(
          elaborate_attention_tile({ request: input, plan, }),
        ),
      ) catch {
        // Enough independent query work can make a two-launch plan illegal.
        InvalidAttentionTileCompilation(ScheduleDerivation) => continue
        error => raise error
      }
      if attention_tile_resource_score(input, compilation) ==
        9223372036854775807L {
        continue
      }
      if !variants.any(v => v.compilation.digest() == compilation.digest()) {
        variants.push({ parallel_workgroup_target: target, compilation, })
      }
    }
  }
  if variants.is_empty() {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let builder = StringBuilder()
  builder <+ "luna-attention-joint-partitioned-alternatives-v1\n"
  for variant in variants {
    builder <+
      "variant=\{variant.parallel_workgroup_target},\{variant.compilation.digest().as_hex()}\n"
  }
  let canonical = @utf8.encode(builder.to_string())
  {
    variants: ReadOnlyArray::from_array(variants),
    canonical,
    digest: AttentionTileCompilationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}

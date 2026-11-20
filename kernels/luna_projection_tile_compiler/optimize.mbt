///|
fn projection_rewrite_code(rule : ProjectionTileRewriteRule) -> Int32 {
  match rule {
    PruneUnobservedTokenRows => 1
    HoistSelectedInputTile => 2
    ReuseInputTileAcrossOutputTiles => 3
    FuseSiblingInputTraversals => 4
    ReuseIntermediateTileAcrossOutputTiles => 5
    FuseAttentionIngressEpilogue => 6
    ElideProjectedQkvRoundTrip => 7
    HoistRotaryInverseFrequencies => 8
    FuseRotaryPairEvaluation => 9
  }
}

///|
pub fn optimize_projection_tile(
  semantic : SemanticProjectionTileCompilation,
) -> OptimizedProjectionTileCompilation {
  let normalized = normalize_projection_tile(semantic)
  let program = normalized.program
  let selected = program.demand == QueryRowEnds
  let facts = projection_dataflow_facts(program)
  let gated_mlp = facts.gated_intermediate
  // Input reuse is a semantic sharing opportunity only when the
  // functional graph has multiple consumers of the same input value. A lone
  // projection has one Dot consumer; staging its input across output tiles is
  // a backend scheduling candidate whose synchronization cost must be proven,
  // not an unconditional functional rewrite.
  let shared_input_value = facts.shared_dot_input
  let trace = [
    { rule: PruneUnobservedTokenRows, changed: selected, },
    { rule: HoistSelectedInputTile, changed: selected, },
    { rule: ReuseInputTileAcrossOutputTiles, changed: shared_input_value, },
    { rule: FuseSiblingInputTraversals, changed: gated_mlp, },
    { rule: ReuseIntermediateTileAcrossOutputTiles, changed: gated_mlp, },
  ]
  let fused_attention_ingress = facts.exclusive_attention_ingress &&
    program.epilogue is ProjectionAttentionIngress(_)
  if fused_attention_ingress {
    trace.push({ rule: FuseAttentionIngressEpilogue, changed: true, })
    trace.push({ rule: ElideProjectedQkvRoundTrip, changed: true, })
    trace.push({ rule: HoistRotaryInverseFrequencies, changed: true, })
    trace.push({ rule: FuseRotaryPairEvaluation, changed: true, })
  }
  let trace = ReadOnlyArray::from_array(trace)
  let builder = StringBuilder()
  builder <+ "luna-projection-tile-optimization-v3\n"
  builder <+ "program_sha256=\{program.digest.0}\n"
  for step in trace {
    builder <+ "pass=\{projection_rewrite_code(step.rule)},\{step.changed}\n"
  }
  let canonical = @utf8.encode(builder.to_string())
  let optimization = ProjectionTileOptimization::{
    program_digest: program.digest,
    trace,
    canonical,
    digest: ProjectionTileOptimizationDigest(projection_tile_sha256(canonical)),
  }
  { semantic, normalized, optimization, }
}

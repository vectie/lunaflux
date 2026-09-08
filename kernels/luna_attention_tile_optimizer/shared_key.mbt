///|
/// map(q, fold(d, dot(q[d], k[d]))) =
/// fold(d, map(q, dot(q[d], k[d]))) over a product of accumulators.
/// Query accumulators are independent, while the key input is identical.
/// This exchanges map and fold; it does not exchange reduction elements.
fn apply_shared_key_query_map_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> AttentionRewritePassResult {
  match program.candidate().compute() {
    AttentionMatrixDot | AttentionMatrixQkPv => ()
    _ => return { regions, changed: false }
  }
  for binding in program.bindings() {
    match binding.expression() {
      AttentionTileDot(query, key) => {
        let shared_key = match find_binding(program, key) {
          Some(key_binding) =>
            key_binding.expression() == AttentionTileInput(PagedKeyTileStream)
          None => false
        }
        if !shared_key ||
          contains_region_root(
            regions,
            FactorSharedKeyAcrossQueryMap,
            binding.id(),
          ) {
          return { regions, changed: false }
        }
        return {
          regions: append_region(regions, {
            rule: FactorSharedKeyAcrossQueryMap,
            members: ReadOnlyArray::from_array([query, key, binding.id()]),
            root: binding.id(),
          }),
          changed: true,
        }
      }
      _ => ()
    }
  }
  { regions, changed: false }
}

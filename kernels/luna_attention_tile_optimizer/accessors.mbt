///|
pub fn AttentionTileRewriteRegion::rule(
  self : AttentionTileRewriteRegion,
) -> AttentionTileRewriteRule {
  self.rule
}

///|
pub fn AttentionTileRewriteRegion::members(
  self : AttentionTileRewriteRegion,
) -> ReadOnlyArray[@luna_attention_tile_ir.AttentionTileValueId] {
  self.members
}

///|
pub fn AttentionTileRewriteRegion::root(
  self : AttentionTileRewriteRegion,
) -> @luna_attention_tile_ir.AttentionTileValueId {
  self.root
}

///|
pub fn AttentionTileRewriteStep::rule(
  self : AttentionTileRewriteStep,
) -> AttentionTileRewriteRule {
  self.rule
}

///|
pub fn AttentionTileRewriteStep::changed(
  self : AttentionTileRewriteStep,
) -> Bool {
  self.changed
}

///|
pub fn AttentionTileRewriteStep::region_count(
  self : AttentionTileRewriteStep,
) -> Int {
  self.region_count
}

///|
pub fn AttentionTileOptimization::program_digest(
  self : AttentionTileOptimization,
) -> @luna_attention_tile_ir.AttentionTileProgramDigest {
  self.program_digest
}

///|
pub fn AttentionTileOptimization::regions(
  self : AttentionTileOptimization,
) -> ReadOnlyArray[AttentionTileRewriteRegion] {
  self.regions
}

///|
pub fn AttentionTileOptimization::trace(
  self : AttentionTileOptimization,
) -> ReadOnlyArray[AttentionTileRewriteStep] {
  self.trace
}

///|
pub fn AttentionTileOptimization::canonical_bytes(
  self : AttentionTileOptimization,
) -> Bytes {
  self.canonical
}

///|
pub fn AttentionTileOptimization::digest(
  self : AttentionTileOptimization,
) -> AttentionTileOptimizationDigest {
  self.digest
}

///|
pub fn AttentionTileOptimizationDigest::as_hex(
  self : AttentionTileOptimizationDigest,
) -> String {
  self.0
}

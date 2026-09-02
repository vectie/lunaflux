///|
/// Backend-neutral equational rewrites recognized by the attention optimizer.
/// The names describe semantic regions, never device instructions.
pub(all) enum AttentionTileRewriteRule {
  FuseScoreTransform
  FuseOnlineSoftmaxFold
} derive(Debug, Eq)

///|
/// One immutable region selected by a semantics-preserving rewrite.
pub struct AttentionTileRewriteRegion {
  priv rule : AttentionTileRewriteRule
  priv members : ReadOnlyArray[@luna_attention_tile_ir.AttentionTileValueId]
  priv root : @luna_attention_tile_ir.AttentionTileValueId
} derive(Debug, Eq)

///|
/// A trace entry makes pure pass composition observable and reproducible.
pub struct AttentionTileRewriteStep {
  priv rule : AttentionTileRewriteRule
  priv changed : Bool
  priv region_count : Int
} derive(Debug, Eq)

///|
pub struct AttentionTileOptimizationDigest(String) derive(Debug, Eq, Hash)

///|
/// Persistent optimization result. The semantic program remains the source of
/// truth; rewrite regions are an immutable extraction plan for every backend.
pub struct AttentionTileOptimization {
  priv program_digest : @luna_attention_tile_ir.AttentionTileProgramDigest
  priv regions : ReadOnlyArray[AttentionTileRewriteRegion]
  priv trace : ReadOnlyArray[AttentionTileRewriteStep]
  priv canonical : Bytes
  priv digest : AttentionTileOptimizationDigest
} derive(Debug, Eq)

///|
pub(all) enum AttentionTileOptimizerRule {
  InvalidRewritePipeline
} derive(Debug, Eq)

///|
pub(all) suberror AttentionTileOptimizerError {
  InvalidAttentionTileOptimization(AttentionTileOptimizerRule)
} derive(Debug, Eq)

///|
pub(all) enum AttentionTileInputRole {
  QueryTileStream
  PagedKeyTileStream
  PagedValueTileStream
} derive(Debug, Eq)

///|
pub(all) enum AttentionTileValueKind {
  QueryTileValue
  KeyTileValue
  ValueTileValue
  ScoreTileValue
  OnlineSoftmaxStateValue
  OutputTileValue
} derive(Debug, Eq)

///|
pub struct AttentionTileValueId(Int) derive(Debug, Eq, Hash)

///|
pub(all) enum AttentionTileExpression {
  AttentionTileInput(AttentionTileInputRole)
  AttentionTileDot(AttentionTileValueId, AttentionTileValueId)
  AttentionTileScale(AttentionTileValueId)
  AttentionTileCausalMask(AttentionTileValueId)
  AttentionTileOnlineSoftmax(
    scores~ : AttentionTileValueId,
    values~ : AttentionTileValueId
  )
  AttentionTileFoldKeyValueStream(AttentionTileValueId)
  AttentionTileNormalize(AttentionTileValueId)
} derive(Debug, Eq)

///|
pub struct AttentionTileBinding {
  priv id : AttentionTileValueId
  priv kind : AttentionTileValueKind
  priv expression : AttentionTileExpression
} derive(Debug, Eq)

///|
pub struct AttentionTileProgramDigest(String) derive(Debug, Eq, Hash)

///|
/// Immutable functional attention tile program. Every binding is a pure value;
/// the only recurrence is the explicit online-softmax fold combinator.
pub struct AttentionTileProgram {
  priv problem : @luna_attention_strategy.AttentionCompileProblem
  priv candidate : @luna_attention_strategy.AttentionTileCandidate
  priv bindings : ReadOnlyArray[AttentionTileBinding]
  priv root : AttentionTileValueId
  priv canonical : Bytes
  priv digest : AttentionTileProgramDigest
} derive(Debug, Eq)

///|
pub(all) enum AttentionTileIrRule {
  InvalidValueId
  InvalidTopology
  InvalidValueKind
  InvalidRoot
} derive(Debug, Eq)

///|
pub(all) suberror AttentionTileIrError {
  InvalidAttentionTileIr(AttentionTileIrRule)
} derive(Debug, Eq)

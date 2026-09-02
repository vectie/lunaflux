///|
pub fn AttentionTileValueId::from_stable_int(
  value : Int,
) -> AttentionTileValueId raise AttentionTileIrError {
  if value <= 0 || value > 4096 {
    raise InvalidAttentionTileIr(InvalidValueId)
  }
  AttentionTileValueId(value)
}

///|
pub fn AttentionTileValueId::as_int(self : AttentionTileValueId) -> Int {
  self.0
}

///|
pub fn AttentionTileBinding::id(
  self : AttentionTileBinding,
) -> AttentionTileValueId {
  self.id
}

///|
pub fn AttentionTileBinding::kind(
  self : AttentionTileBinding,
) -> AttentionTileValueKind {
  self.kind
}

///|
pub fn AttentionTileBinding::expression(
  self : AttentionTileBinding,
) -> AttentionTileExpression {
  self.expression
}

///|
pub fn AttentionTileProgram::problem(
  self : AttentionTileProgram,
) -> @luna_attention_strategy.AttentionCompileProblem {
  self.problem
}

///|
pub fn AttentionTileProgram::candidate(
  self : AttentionTileProgram,
) -> @luna_attention_strategy.AttentionTileCandidate {
  self.candidate
}

///|
pub fn AttentionTileProgram::bindings(
  self : AttentionTileProgram,
) -> ReadOnlyArray[AttentionTileBinding] {
  self.bindings
}

///|
pub fn AttentionTileProgram::root(
  self : AttentionTileProgram,
) -> AttentionTileValueId {
  self.root
}

///|
pub fn AttentionTileProgram::canonical_bytes(
  self : AttentionTileProgram,
) -> Bytes {
  self.canonical
}

///|
pub fn AttentionTileProgram::digest(
  self : AttentionTileProgram,
) -> AttentionTileProgramDigest {
  self.digest
}

///|
pub fn AttentionTileProgramDigest::as_hex(
  self : AttentionTileProgramDigest,
) -> String {
  self.0
}

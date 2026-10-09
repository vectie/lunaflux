///|
pub fn GlmTinyHybridBlockResult::attention_branch(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.attention_branch
}

///|
pub fn GlmTinyHybridBlockResult::feed_forward_branch(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.feed_forward_branch
}

///|
pub fn GlmTinyHybridBlockResult::output(self : Self) -> ReadOnlyArray[Double] {
  self.output
}

///|
pub fn GlmTinyHybridBlockResult::selected_experts(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.selected_experts
}

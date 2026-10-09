///|
pub fn GlmKdaProjectionResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmKdaProjectionResult::projected_width(self : Self) -> Int {
  self.projected_width
}

///|
pub fn GlmKdaProjectionResult::control_rank(self : Self) -> Int {
  self.control_rank
}

///|
pub fn GlmKdaProjectionResult::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmKdaProjectionResult::query(self : Self) -> ReadOnlyArray[Double] {
  self.query
}

///|
pub fn GlmKdaProjectionResult::key(self : Self) -> ReadOnlyArray[Double] {
  self.key
}

///|
pub fn GlmKdaProjectionResult::value(self : Self) -> ReadOnlyArray[Double] {
  self.value
}

///|
pub fn GlmKdaProjectionResult::forget_low_rank(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.forget_low_rank
}

///|
pub fn GlmKdaProjectionResult::raw_forget(self : Self) -> ReadOnlyArray[Double] {
  self.raw_forget
}

///|
pub fn GlmKdaProjectionResult::beta(self : Self) -> ReadOnlyArray[Double] {
  self.beta
}

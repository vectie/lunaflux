///|
pub fn ExecutionGraphBucket::slot(self : ExecutionGraphBucket) -> Int {
  self.slot
}

///|
pub fn ExecutionGraphBucket::phase(
  self : ExecutionGraphBucket,
) -> ExecutionGraphPhase {
  self.phase
}

///|
pub fn ExecutionGraphBucket::batch_rows(self : ExecutionGraphBucket) -> Int {
  self.batch_rows
}

///|
pub fn ExecutionGraphBucket::query_tokens(self : ExecutionGraphBucket) -> Int {
  self.query_tokens
}

///|
pub fn ExecutionGraphBucket::context_tokens(self : ExecutionGraphBucket) -> Int {
  self.context_tokens
}

///|
pub fn ExecutionGraphBucket::upper_bounds(
  self : ExecutionGraphBucket,
) -> (Int, Int, Int) {
  (self.batch_rows, self.query_tokens, self.context_tokens)
}

///|
pub fn ExecutionGraphBucketTable::bucket_count(
  self : ExecutionGraphBucketTable,
) -> Int {
  self.bucket_count
}

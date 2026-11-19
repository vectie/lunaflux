///|
pub fn ExecutionGraphBucket::slot(self : ExecutionGraphBucket) -> Int32 {
  self.slot
}

///|
pub fn ExecutionGraphBucket::phase(
  self : ExecutionGraphBucket,
) -> ExecutionGraphPhase {
  self.phase
}

///|
pub fn ExecutionGraphBucket::batch_rows(self : ExecutionGraphBucket) -> Int32 {
  self.batch_rows
}

///|
pub fn ExecutionGraphBucket::query_tokens(self : ExecutionGraphBucket) -> Int32 {
  self.query_tokens
}

///|
pub fn ExecutionGraphBucket::context_tokens(
  self : ExecutionGraphBucket,
) -> Int32 {
  self.context_tokens
}

///|
pub fn ExecutionGraphBucket::upper_bounds(
  self : ExecutionGraphBucket,
) -> (Int32, Int32, Int32) {
  (self.batch_rows, self.query_tokens, self.context_tokens)
}

///|
pub fn ExecutionGraphBucketTable::bucket_count(
  self : ExecutionGraphBucketTable,
) -> Int32 {
  self.bucket_count
}

///|
fn graph_bucket_exponent(value : Int) -> Int raise ExecutionGraphBucketError {
  if value <= 0 || value > 1048576 {
    raise InvalidExecutionGraphBucket(Shape)
  }
  // ceil(log2(value)); clz(0) is the Int bit width, so one maps to zero.
  32 - (value - 1).clz()
}

///|
fn graph_bucket_value(exponent : Int) -> Int {
  1 << exponent
}

///|
/// First value in the following bucket. The result may be one past the
/// supported domain, allowing bounded traversals to terminate without a
/// special final bucket. No value in (value, result) selects a different bucket.
pub fn next_execution_graph_bucket_start(
  value : Int,
) -> Int raise ExecutionGraphBucketError {
  graph_bucket_value(graph_bucket_exponent(value)) + 1
}

///|
/// Startup inverse of the compact slot mapping; useful for preparing owners.
pub fn ExecutionGraphBucketTable::bucket_at(
  self : ExecutionGraphBucketTable,
  slot : Int,
) -> ExecutionGraphBucket raise ExecutionGraphBucketError {
  if slot < 0 || slot >= self.bucket_count {
    raise InvalidExecutionGraphBucket(Limit)
  }
  let phase = match slot / self.phase_stride {
    0 => PrefillGraph
    1 => DecodeGraph
    _ => MixedGraph
  }
  let offset = slot % self.phase_stride
  {
    slot,
    phase,
    batch_rows: graph_bucket_value(offset / self.query_stride),
    query_tokens: graph_bucket_value(
      offset % self.query_stride / self.context_stride,
    ),
    context_tokens: graph_bucket_value(offset % self.context_stride),
  }
}

///|
fn graph_phase_index(phase : ExecutionGraphPhase) -> Int {
  match phase {
    PrefillGraph => 0
    DecodeGraph => 1
    MixedGraph => 2
  }
}

///|
pub fn ExecutionGraphShape::new(
  phase~ : ExecutionGraphPhase,
  batch_rows~ : Int,
  query_tokens~ : Int,
  context_tokens~ : Int,
) -> ExecutionGraphShape raise ExecutionGraphBucketError {
  let _ = graph_bucket_exponent(batch_rows)
  let _ = graph_bucket_exponent(query_tokens)
  let _ = graph_bucket_exponent(context_tokens)
  { phase, batch_rows, query_tokens, context_tokens, }
}

///|
pub fn build_execution_graph_bucket_table(
  maximum_batch_rows : Int,
  maximum_query_tokens : Int,
  maximum_context_tokens : Int,
  maximum_buckets : Int,
) -> ExecutionGraphBucketTable raise ExecutionGraphBucketError {
  let maximum_batch_exponent = graph_bucket_exponent(maximum_batch_rows)
  let maximum_query_exponent = graph_bucket_exponent(maximum_query_tokens)
  let maximum_context_exponent = graph_bucket_exponent(maximum_context_tokens)
  let context_stride = maximum_context_exponent + 1
  let query_stride = (maximum_query_exponent + 1) * context_stride
  let phase_stride = (maximum_batch_exponent + 1) * query_stride
  let bucket_count = 3 * phase_stride
  if maximum_buckets <= 0 || bucket_count > maximum_buckets {
    raise InvalidExecutionGraphBucket(Capacity)
  }
  {
    maximum_batch_exponent,
    maximum_query_exponent,
    maximum_context_exponent,
    phase_stride,
    query_stride,
    context_stride,
    bucket_count,
  }
}

///|
/// Constant-bounded exponent mapping plus one array lookup; no scan/allocation.
pub fn ExecutionGraphBucketTable::select(
  self : ExecutionGraphBucketTable,
  shape : ExecutionGraphShape,
) -> ExecutionGraphBucket raise ExecutionGraphBucketError {
  let batch = graph_bucket_exponent(shape.batch_rows)
  let query = graph_bucket_exponent(shape.query_tokens)
  let context = graph_bucket_exponent(shape.context_tokens)
  if batch > self.maximum_batch_exponent ||
    query > self.maximum_query_exponent ||
    context > self.maximum_context_exponent {
    raise InvalidExecutionGraphBucket(Limit)
  }
  let slot = graph_phase_index(shape.phase) * self.phase_stride +
    batch * self.query_stride +
    query * self.context_stride +
    context
  {
    slot,
    phase: shape.phase,
    batch_rows: graph_bucket_value(batch),
    query_tokens: graph_bucket_value(query),
    context_tokens: graph_bucket_value(context),
  }
}

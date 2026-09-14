///|
/// Startup traversal of the neutral bucket equivalence classes, not every
/// concrete shape. Keep error translation at the runtime package boundary.
fn next_paged_graph_bucket_start(value : Int) -> Int raise DeviceStepError {
  @luna_execution_graph_strategy.next_execution_graph_bucket_start(value) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
}

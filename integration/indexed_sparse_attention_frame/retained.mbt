///|
/// Bind the retained history and prepend its reserve/copy/publication effects
/// to pooling/index/attention in the caller's one ordered queue. Projection is
/// still an upstream borrowed frame; this does not manufacture model weights.
pub fn IndexedSparseAttentionFrame::prepare_retained(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  history : IndexedSparseHistory,
  maximum_workspace_bytes~ : Int64,
  query_counts~ : @device.Allocation,
  index_queries~ : @device.Allocation,
  index_head_weights~ : @device.Allocation,
  pool_ape~ : @device.Allocation,
  queries~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  let expected = self.source.plan()
  let retained = history.source.plan().attention()
  guard self.status is Fresh &&
    history.status is Prepared &&
    history.counts is Some(bound_counts) &&
    bound_counts.same_storage(query_counts) &&
    expected.capacity() == retained.capacity() &&
    expected.index_geometry() == retained.index_geometry() &&
    expected.attention_geometry() == retained.attention_geometry() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let buffers = history.buffers
  self.prepare(
    context,
    aot,
    maximum_workspace_bytes~,
    query_counts~,
    history_counts=buffers[5],
    query_positions=buffers[7],
    index_queries~,
    index_head_weights~,
    index_keys=buffers[0],
    pool_gate_scores=buffers[1],
    pool_ape~,
    history_valid=buffers[4],
    queries~,
    keys=buffers[2],
    values=buffers[3],
    output~,
  )
  let attention = self.launches[:].to_owned()
  self.launches.clear()
  self.launches.append(history.launches)
  self.launches.append(attention)
}

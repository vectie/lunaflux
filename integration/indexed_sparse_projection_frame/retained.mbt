///|
/// Prepare borrowed projection outputs as history append inputs, then attention
/// inputs. Returns one caller-owned launch list: reserve -> twelve projections
/// -> append/publish -> pool/index/attention. No second queue or completion.
pub fn IndexedSparseProjectionFrame::prepare_retained_attention(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  history : @sparse.IndexedSparseHistory,
  attention : @sparse.IndexedSparseAttentionFrame,
  maximum_state_bytes~ : Int64,
  maximum_attention_workspace_bytes~ : Int64,
  reset_flag~ : @device.Allocation,
  pool_ape~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared && self.counts is Some(query_counts) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let expected = self.source.plan().attention()
  for plan in [history.attention_plan(), attention.attention_plan()] {
    guard plan.capacity() == expected.capacity() &&
      plan.index_geometry() == expected.index_geometry() &&
      plan.attention_geometry() == expected.attention_geometry() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  let outputs = self.projected_outputs()
  history.prepare(
    context,
    aot,
    maximum_state_bytes~,
    query_counts~,
    reset_flag~,
    index_keys=outputs.index_keys,
    pool_gate_scores=outputs.pool_gate_scores,
    keys=outputs.keys,
    values=outputs.values,
  )
  attention.prepare_retained(
    context,
    aot,
    history,
    maximum_workspace_bytes=maximum_attention_workspace_bytes,
    query_counts~,
    index_queries=outputs.index_queries,
    index_head_weights=outputs.index_head_weights,
    pool_ape~,
    queries=outputs.queries,
    output~,
  )
  let retained = attention.launches()
  let launches = [retained[0]]
  for launch in self.launches {
    launches.push(launch)
  }
  for launch in retained[1:] {
    launches.push(launch)
  }
  launches
}

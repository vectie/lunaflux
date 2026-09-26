///|
pub fn PagedGraphExecutor::prepare_streaming(
  self : PagedGraphExecutor,
  capability : @streaming_ir.Capability,
  worker : UInt64,
) -> Unit raise DeviceStepError {
  let descriptor = complete_descriptor(self.resources)
  let separate = descriptor.context.has_separate_host_memory() catch {
    error => raise ExecutorDevice(PersistentKvArena, error)
  }
  if !separate {
    raise InvalidExecutor(ResourceState)
  }
  if worker == 0UL ||
    self.resources.streaming is Some(_) ||
    capability.page_count() != descriptor.layout.total_page_count() {
    raise InvalidExecutor(ResourceState)
  }
  guard self.resources.activation_workspace is Some(workspace) && workspace.open else {
    raise InvalidExecutor(ResourceState)
  }
  if capability.workspace_budget() > workspace.resource.byte_count() {
    raise InvalidExecutor(ResourceState)
  }
  for i in 0..<capability.shape_count() {
    if capability.shape_at(i).tokens() > descriptor.limits.max_tokens {
      raise InvalidExecutor(ResourceState)
    }
  }
  let plan = @host_cache.Plan::new(
    descriptor.layout,
    slots=capability.slots(),
    transfer_lanes=1,
    host_budget=capability.host_bytes(),
    separate_host_memory=true,
  ) catch {
    _ => raise InvalidExecutor(ResourceState)
  }
  if plan.slot_bytes() != capability.page_bytes() {
    raise InvalidExecutor(ResourceState)
  }
  self.prepare_host_cache(plan)
  guard self.resources.host_cache is Some(cache) else {
    raise InvalidExecutor(ResourceState)
  }
  self.resources.streaming = Some(
    StreamingSession::new(capability, worker, cache),
  )
}

///|
pub fn PagedGraphExecutor::streaming_command(
  self : PagedGraphExecutor,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  guard self.resources.streaming is Some(session) else {
    raise InvalidExecutor(ResourceState)
  }
  errdefer {
    self.lifecycle = GraphPoisoned
  }
  session.command(command)
}

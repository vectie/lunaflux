///|
fn require_host_cache_idle(
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  if resources.host_cache is Some(cache) && !cache.is_idle() {
    raise InvalidExecutor(ResourceState)
  }
}

///|
/// Opt-in startup preparation for a worker-local cold-page payload tier. The
/// caller's admitted plan must match the executor's complete KV layout. No
/// mutation of captured graph addresses or kernel arguments occurs.
pub fn PagedGraphExecutor::prepare_host_cache(
  self : PagedGraphExecutor,
  plan : @host_cache.Plan,
) -> Unit raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  let descriptor = complete_descriptor(self.resources)
  if descriptor.stage_epoch != 0UL ||
    self.resources.host_cache is Some(_) ||
    plan.layout() != descriptor.layout {
    raise InvalidExecutor(ResourceState)
  }
  guard self.resources.persistent_kv is Some(allocation) && allocation.open else {
    raise InvalidExecutor(ResourceState)
  }
  let cache = @host_cache.Cache::prepare(
    descriptor.context,
    allocation.resource,
    plan,
  ) catch {
    error => raise ExecutorDevice(PersistentKvArena, error)
  }
  // Publish cleanup authority before inspecting the native preparation result.
  self.resources.host_cache = Some(cache)
  if !cache.is_prepared() {
    self.lifecycle = GraphPoisoned
    raise InvalidExecutor(ResourceState)
  }
}

///|
/// Caller retains the exact scheduler-owned page reservation until completion.
/// Only immutable cold pages may be submitted by the prefix integration.
pub fn PagedGraphExecutor::begin_prefix_spill(
  self : PagedGraphExecutor,
  page_index : Int,
) -> @host_cache.Start raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  guard self.resources.host_cache is Some(cache) else {
    raise InvalidExecutor(ResourceState)
  }
  cache.begin_spill(page_index) catch {
    error => {
      self.lifecycle = GraphPoisoned
      raise ExecutorDevice(PersistentKvArena, error)
    }
  }
}

///|
pub fn PagedGraphExecutor::begin_prefix_restore(
  self : PagedGraphExecutor,
  entry : @residency.Entry,
  page_index : Int,
) -> @host_cache.Start raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  guard self.resources.host_cache is Some(cache) else {
    raise InvalidExecutor(ResourceState)
  }
  cache.begin_restore(entry, page_index) catch {
    error => {
      self.lifecycle = GraphPoisoned
      raise ExecutorDevice(PersistentKvArena, error)
    }
  }
}

///|
pub fn PagedGraphExecutor::poll_prefix_transfer(
  self : PagedGraphExecutor,
  operation : @host_cache.Operation,
) -> @host_cache.Progress raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  guard self.resources.host_cache is Some(cache) else {
    raise InvalidExecutor(ResourceState)
  }
  cache.poll(operation) catch {
    error => {
      self.lifecycle = GraphPoisoned
      raise ExecutorDevice(PersistentKvArena, error)
    }
  }
}

///|
pub fn PagedGraphExecutor::cancel_prefix_transfer(
  self : PagedGraphExecutor,
  operation : @host_cache.Operation,
) -> Bool {
  match self.resources.host_cache {
    Some(cache) => cache.cancel(operation)
    None => false
  }
}

///|
pub fn PagedGraphExecutor::evict_host_prefix(
  self : PagedGraphExecutor,
  entry : @residency.Entry,
) -> Bool {
  match self.resources.host_cache {
    Some(cache) => cache.evict(entry)
    None => false
  }
}

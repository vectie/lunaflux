///|
priv struct TensorParallelStreaming {
  cache : @host_cache.Cache
  session : @device_step.StreamingSession
}

///|
fn prepare_streaming(
  plan : TensorParallelDeviceWorkerPlan,
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard plan.startup.streaming() is Some(capability) else { return }
  let context = require_context(resources)
  let kv = require_kv(resources)
  let separate = context.has_separate_host_memory() catch {
    error => raise Device(PersistentKv, error)
  }
  if !separate ||
    capability.page_count() != kv.layout().total_page_count() ||
    capability.workspace_budget() > plan.execution.activation_workspace_bytes() ||
    plan.startup.streaming_epoch() != plan.group_generation.value() {
    raise Invalid(KvBinding)
  }
  for index in 0..<capability.shape_count() {
    if capability.shape_at(index).tokens() > plan.descriptor_limits.max_tokens() {
      raise Invalid(ProfileEnvelope)
    }
  }
  let cache_plan = @host_cache.Plan::new(
    kv.layout(),
    slots=capability.slots(),
    transfer_lanes=1,
    host_budget=capability.host_bytes(),
    separate_host_memory=true,
  ) catch {
    _ => raise Invalid(KvBinding)
  }
  if cache_plan.slot_bytes() != capability.page_bytes() {
    raise Invalid(KvBinding)
  }
  let cache = kv.prepare_host_cache(context, cache_plan) catch {
    error => raise Kv(error)
  }
  resources.streaming = Some({
    cache,
    session: @device_step.StreamingSession::new(
      capability,
      plan.startup.streaming_epoch(),
      cache,
    ),
  })
  if !cache.is_prepared() {
    raise Invalid(ResourceState)
  }
}

///|
fn close_streaming(
  streaming : TensorParallelStreaming?,
) -> Unit raise TensorParallelDeviceWorkerError {
  if streaming is Some(streaming) {
    streaming.cache.close() catch {
      error => raise Device(PersistentKv, error)
    }
  }
}

///|
pub fn TensorParallelDeviceWorkerOwner::streaming_command(
  self : TensorParallelDeviceWorkerOwner,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise TensorParallelDeviceWorkerError {
  if self.lifecycle != WorkerReady {
    raise Invalid(ResourceState)
  }
  guard self.resources.streaming is Some(streaming) else {
    raise Invalid(KvBinding)
  }
  streaming.session.command(command) catch {
    error => raise fail_rank_execution(self, Descriptor(error))
  }
}

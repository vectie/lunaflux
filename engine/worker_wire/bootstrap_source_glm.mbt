///|
/// GLM's manifest describes a complete batched TP owner, not a dense paged
/// Llama worker. This version also requires GPU full-F32 sampling and exact
/// request retirement receipts. Labels are supplied, never computed here.
pub fn admit_glm_batched_bootstrap_source(
  roots : WorkerBootstrapRoots,
  model : WorkerBootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerBootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
) -> EncodedBootstrapSource raise WorkerWireError {
  guard kernel.luna_approval is None &&
    kernel.fused_runtime is None &&
    kernel.sampling_runtime == DeviceFullF32SamplingV1 else {
    raise InvalidBootstrapSource(Recipe)
  }
  // Fresh unpublished storage; retain the common bounded locator envelope.
  let base = admit_bootstrap_source(
    roots,
    model,
    { ..kernel, sampling_runtime: HostSamplingRuntime, },
    execution,
    ceilings,
  )
  write_u32(base.storage, 4, 14U)
  write_i32(base.storage, 8, 14)
  { ..base, recipe: PackedGlmBatchedTensorV14, kernel, }
}

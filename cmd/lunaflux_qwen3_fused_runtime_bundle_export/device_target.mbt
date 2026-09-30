///|
/// Legacy direct invocations retain their original sm120 contract. The recipe
/// builder always supplies the compiled target explicitly, including sm121.
fn runtime_device_target(
  args : RuntimeBundleArguments,
) -> @catalog.DeviceTarget raise Qwen3FusedRuntimeBundleExportError {
  match args.device_target {
    Some(target) => target
    None =>
      @catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=0,
        supports_bf16=true,
        supports_cublas_lt=false,
      ) catch {
        _ => raise InvalidRuntime
      }
  }
}

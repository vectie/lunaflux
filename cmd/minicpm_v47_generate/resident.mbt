///|
/// Shared startup and teardown for offline generation and chat serving.
async fn with_resident(
  model : @bootstrap.PreparedModel,
  root : @approved_fs.ApprovedRoot,
  kernel_directory : String,
  device : Int32,
  maximum_device_bytes : Int64,
  body : async (@runtime.Resident, @device.DeviceCapability) -> Unit,
) -> Unit {
  guard device >= 0 && maximum_device_bytes > 0L else {
    fail("invalid execution limits")
  }
  let kernels = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(kernel_directory),
  )
  defer kernels.close()
  let images = @runtime.ModuleImages::read(
    kernels,
    maximum_module_bytes=67108864L,
  )
  @async.with_task_group() <| group => {
    let context = @device.open_context(device)
    group.add_defer(() => context.close())
    let facts = context.capability()
    guard facts.compute_major() == 12 &&
      facts.compute_minor() == 1 &&
      facts.supports_bf16() &&
      maximum_device_bytes <= facts.total_memory_bytes() else {
      fail(
        "resident profile requires BF16 sm121 and a bounded device allocation ceiling",
      )
    }
    let resident = @runtime.Resident::new(model.program, model.weights)
    group.add_defer(() => resident.close())
    resident.prepare(
      context,
      root,
      images,
      maximum_device_bytes~,
      scratch=FixedArray::make(4194304, b'\x00'),
      limits=@bootstrap.streaming_limits(),
    )
    body(resident, facts)
  }
}

///|
async fn serve(
  model_directory : String,
  kernel_directory : String,
  bind_address : String,
  device : Int32,
  maximum_device_bytes : Int64,
) -> Unit {
  // The existing bounded HTTP ingress requires a configured bearer policy.
  // Read it before preparing model/device resources; never log its bytes.
  let policy = chat_policy()
  defer policy.close()
  let timeout_ms = chat_timeout()
  let address = @socket.Addr::parse(bind_address)
  let root = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(model_directory),
  )
  defer root.close()
  let model = @bootstrap.prepare(root)
  with_resident(model, root, kernel_directory, device, maximum_device_bytes, (
    resident,
    _,
  ) => {
    let backend = @chat.Backend::new(
      model.model,
      model.tokenizer,
      resident.runner(),
    )
    let service = backend.service(policy, timeout_ms~)
    defer service.close()
    let listener = @socket.TcpServer(address)
    defer listener.close()
    println(
      "MiniCPM-V 4.7 buffered chat listening: " + listener.addr.to_string(),
    )
    service.serve(listener)
  })
}

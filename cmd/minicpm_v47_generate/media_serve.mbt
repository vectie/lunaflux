///|
async fn serve_media(
  model_directory : String,
  kernel_directory : String,
  bind_address : String,
  device : Int32,
  maximum_device_bytes : Int64,
) -> Unit {
  let policy = chat_policy()
  defer policy.close()
  let timeout_ms = chat_timeout()
  let address = @socket.Addr::parse(bind_address)
  let root = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(model_directory),
  )
  defer root.close()
  let model = @bootstrap.prepare_images(
    root,
    maximum_patches=16384,
    maximum_crop_patches=8192,
  )
  let program = @multimodal.Program::new(model.text.program)
  with_image_resident(
    model,
    program,
    root,
    kernel_directory,
    device,
    maximum_device_bytes,
    (resident, _) => {
      let backend = @chat.Backend::with_images(
        model.text.model,
        model.text.tokenizer,
        model.vision,
        resident.execution(),
      )
      let service = backend.service(policy, timeout_ms~)
      defer service.close()
      let listener = @socket.TcpServer(address)
      defer listener.close()
      println(
        "MiniCPM-V 4.7 buffered image chat listening: " +
        listener.addr.to_string(),
      )
      service.serve(listener)
    },
  )
}

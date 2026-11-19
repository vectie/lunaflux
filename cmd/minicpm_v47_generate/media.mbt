///|
async fn media_mode(
  model_directory : String,
  kernel_directory : String?,
  media_directory : String,
  queue_file : String,
  output : String,
  device : Int32,
  maximum_device_bytes : Int64,
) -> Unit {
  guard !@fs.exists(output) else { fail("output must be a new directory") }
  let root = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(model_directory),
  )
  defer root.close()
  let media_root = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(media_directory),
  )
  defer media_root.close()
  let model = @bootstrap.prepare_images(
    root,
    maximum_patches=16384,
    maximum_crop_patches=8192,
  )
  let program = @multimodal.Program::new(model.text.program)
  let queue = @fs.open(queue_file, mode=ReadOnly)
  defer queue.close()
  let bytes = queue.size()
  guard bytes > 0L && bytes <= 1048576L else {
    fail("request file exceeds media queue budget")
  }
  let requests = @media.prepare_requests(
    model.text.model,
    model.text.tokenizer,
    model.vision,
    @json_guard.parse_bounded(
      @utf8.decode(queue.read_exactly(bytes.to_int())),
      max_depth=32,
    ),
    load_image=path => {
      let file = media_root.open_file(
        @approved_fs.ApprovedRelativeLocator::new(path),
      )
      defer file.close()
      @rgb8.Image::decode_oriented(
        file.read_immutable_snapshot(maximum_bytes=33554432L),
        maximum_encoded_bytes=33554432,
        maximum_pixels=16777216,
      )
    },
    maximum_prepared_pixel_bytes=268435456L,
  )
  let budget = @image_runtime.Budget::new(
    program,
    model.vision,
    text_weight_bytes=model.text.weights.byte_count(),
    vision_weight_bytes=model.vision_weights.byte_count(),
  )
  @fs.mkdir(output, recursive=true)
  write_receipt(
    output,
    "plan.json",
    media_plan_receipt(model, requests, budget),
  )
  if kernel_directory is Some(directory) {
    with_image_resident(
      model,
      program,
      root,
      directory,
      device,
      maximum_device_bytes,
      (resident, facts) => {
        write_receipt(output, "startup.json", {
          "format": "lunaflux.minicpm-v47-media-startup.v1",
          "device": Json(device),
          "observed_compute_major": Json(facts.compute_major()),
          "observed_compute_minor": Json(facts.compute_minor()),
          "observed_total_memory_bytes": Json(facts.total_memory_bytes()),
          "weights_loaded": true,
          "aot_module_count": 9,
          "resident_prepared": true,
          "device_qualified": false,
        })
        let execution = resident.execution()
        let stops = ReadOnlyArray::from_array(@prompt.stop_token_ids())
        for index in 0..<requests.length() {
          let request = requests[index]
          match request.input.vision_input() {
            Some((prompt, input)) => {
              execution.begin_image(
                prompt,
                input,
                maximum_new_tokens=request.maximum_new_tokens,
                stop_token_ids=stops,
                context_ceiling=request.context_ceiling,
              )
              execution.wait()
            }
            None =>
              execution.begin_text(
                request.input.prompt(),
                maximum_new_tokens=request.maximum_new_tokens,
                stop_token_ids=stops,
                context_ceiling=request.context_ceiling,
              )
          }
          while execution.submit_next() {
            execution.wait()
          }
          let tokens = execution.emitted_tokens()
          let (text, valid) = decode_output(model.text.tokenizer, tokens)
          write_receipt(output, "result-\{index}.json", {
            "format": "lunaflux.minicpm-v47-media-result.v1",
            "id": Json(request.id),
            "token_ids": Json(tokens.to_owned()),
            "text": Json(text),
            "text_utf8_valid": Json(valid),
            "input_tokens": Json(request.input.prompt().tokens().length()),
            "image_count": Json(request.input.image_count()),
            "image_feature_rows": Json(request.input.feature_rows()),
            "sampled_tokens_including_stop": Json(execution.sampled_tokens()),
            "emitted_tokens": Json(tokens.length()),
            "cached_positions": Json(execution.cached_tokens()),
            "finish_reason": Json(
              match execution.finish_reason() {
                Some(StopSequence) => "stop"
                Some(TokenLimit) => "length"
                Some(Cancelled) => "cancelled"
                None => "incomplete"
              },
            ),
            "gpu_execution": true,
            "numerical_qualified": false,
            "device_qualified": false,
          })
        }
      },
    )
  }
  println("MiniCPM-V 4.7 media receipts: " + output)
}

///|
async fn with_image_resident(
  model : @bootstrap.PreparedImageModel,
  program : @multimodal.Program,
  root : @approved_fs.ApprovedRoot,
  kernel_directory : String,
  device : Int32,
  maximum_device_bytes : Int64,
  body : async (@image_runtime.Resident, @device.DeviceCapability) -> Unit,
) -> Unit {
  guard device >= 0 && maximum_device_bytes > 0L else {
    fail("invalid execution limits")
  }
  let kernels = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(kernel_directory),
  )
  defer kernels.close()
  let images = @image_runtime.ModuleImages::read(
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
    let resident = @image_runtime.Resident::new(
      program,
      model.vision,
      model.text.weights,
      model.vision_weights,
    )
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

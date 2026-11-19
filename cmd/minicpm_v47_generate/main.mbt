///|
async fn main {
  let args = @env.args()
  match args[1:] {
    ["serve-media", model, kernels, address, device, ceiling] => {
      serve_media(
        model,
        kernels,
        address,
        @string.parse_int(device),
        @string.parse_int64(ceiling),
      )
      return
    }
    ["inspect-media", model, media, queue, output] => {
      media_mode(model, None, media, queue, output, 0, 0L)
      return
    }
    ["run-media", model, kernels, media, queue, output, device, ceiling] => {
      media_mode(
        model,
        Some(kernels),
        media,
        queue,
        output,
        @string.parse_int(device),
        @string.parse_int64(ceiling),
      )
      return
    }
    _ => ()
  }
  if args[1:] is ["serve", model, kernels, address, device, ceiling] {
    serve(
      model,
      kernels,
      address,
      @string.parse_int(device),
      @string.parse_int64(ceiling),
    )
    return
  }
  let (
    model_directory,
    kernel_directory,
    queue_file,
    output,
    device,
    maximum_device_bytes,
  ) = match args[1:] {
    ["inspect", model, queue, output] => (model, None, queue, output, 0, 0L)
    ["run", model, kernels, queue, output, device, ceiling] =>
      (
        model,
        Some(kernels),
        queue,
        output,
        @string.parse_int(device),
        @string.parse_int64(ceiling),
      )
    _ =>
      fail(
        "usage: minicpm_v47_generate inspect MODEL_ROOT REQUESTS_JSON NEW_OUTPUT | run MODEL_ROOT CUBIN_ROOT REQUESTS_JSON NEW_OUTPUT DEVICE MAX_DEVICE_BYTES | serve MODEL_ROOT CUBIN_ROOT IP:PORT DEVICE MAX_DEVICE_BYTES | inspect-media MODEL_ROOT MEDIA_ROOT REQUESTS_JSON NEW_OUTPUT | run-media MODEL_ROOT CUBIN_ROOT MEDIA_ROOT REQUESTS_JSON NEW_OUTPUT DEVICE MAX_DEVICE_BYTES | serve-media MODEL_ROOT CUBIN_ROOT IP:PORT DEVICE MAX_DEVICE_BYTES",
      )
  }
  guard !@fs.exists(output) else { fail("output must be a new directory") }
  let root = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(model_directory),
  )
  defer root.close()
  let model = @bootstrap.prepare(root)
  let queue = @fs.open(queue_file, mode=ReadOnly)
  defer queue.close()
  let queue_bytes = queue.size()
  guard queue_bytes > 0L && queue_bytes <= 1048576L else {
    fail("request file exceeds the text queue budget")
  }
  let requests = @bootstrap.prepare_requests(
    model.model,
    model.tokenizer,
    @json_guard.parse_bounded(
      @utf8.decode(queue.read_exactly(queue_bytes.to_int())),
      max_depth=32,
    ),
  )
  @fs.mkdir(output, recursive=true)
  write_receipt(output, "plan.json", plan_receipt(model, requests))
  if kernel_directory is Some(directory) {
    with_resident(model, root, directory, device, maximum_device_bytes, (
      resident,
      facts,
    ) => {
      write_receipt(output, "startup.json", {
        "format": "lunaflux.minicpm-v47-text-startup.v1",
        "device": Json(device),
        "observed_compute_major": Json(facts.compute_major()),
        "observed_compute_minor": Json(facts.compute_minor()),
        "observed_total_memory_bytes": Json(facts.total_memory_bytes()),
        "weights_loaded": true,
        "aot_modules_loaded": true,
        "resident_prepared": true,
        "device_qualified": false,
      })
      let runner = resident.runner()
      let stops = ReadOnlyArray::from_array(@prompt.stop_token_ids())
      for index in 0..<requests.length() {
        let request = requests[index]
        runner.begin(
          request.prompt,
          maximum_new_tokens=request.maximum_new_tokens,
          stop_token_ids=stops,
          context_ceiling=request.context_ceiling,
        )
        while runner.submit_next() {
          runner.wait()
        }
        // Decoding/string allocation and evidence publication happen after
        // request completion, outside all token-step measurements.
        let tokens = runner.emitted_tokens()
        let (text, text_utf8_valid) = decode_output(model.tokenizer, tokens)
        write_receipt(output, "result-\{index}.json", {
          "format": "lunaflux.minicpm-v47-text-result.v1",
          "id": Json(request.id),
          "token_ids": Json(tokens.to_owned()),
          "text": Json(text),
          "text_utf8_valid": Json(text_utf8_valid),
          "input_tokens": Json(request.prompt.tokens().length()),
          "sampled_tokens_including_stop": Json(runner.sampled_tokens()),
          "emitted_tokens": Json(tokens.length()),
          "cached_positions": Json(runner.cached_tokens()),
          "finish_reason": Json(
            match runner.finish_reason() {
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
    })
  }
  println("MiniCPM-V 4.7 text receipts: " + output)
}

///|
/// Offline device diagnostic. Readbacks and files are deliberately outside
/// production runtime owners and their steady-state token path.
async fn main {
  guard @env.args()[1:] is [model_path, cubin_path, queue_path, output] &&
    !@fs.exists(output) else {
    fail(
      "usage: minicpm_v47_model_oracle MODEL_ROOT CUBIN_ROOT REQUESTS.json NEW_OUTPUT",
    )
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(model_path))
  defer root.close()
  let kernels = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath(cubin_path),
  )
  defer kernels.close()
  let model = @bootstrap.prepare(root)
  let requests = @bootstrap.prepare_requests(
    model.model,
    model.tokenizer,
    @fs.read_file(queue_path).json(),
  )
  let images = @runtime.ModuleImages::read(
    kernels,
    maximum_module_bytes=67108864L,
  )
  @fs.mkdir(output, recursive=true)
  let prompts : Array[Json] = []
  for request in requests {
    prompts.push({
      "id": Json(request.id),
      "token_ids": Json(request.prompt.tokens()),
      "maximum_new_tokens": Json(request.maximum_new_tokens),
    })
  }
  @fs.write_file(
    output + "/plan.json",
    Json({ "requests": Json(prompts) }).stringify(indent=2),
    create_mode=CreateNew,
  )
  @async.with_task_group() <| group => {
    let context = @device.open_context(0)
    group.add_defer(() => context.close())
    let capability = context.capability()
    guard capability.compute_major() == 12 &&
      capability.compute_minor() == 1 &&
      capability.supports_bf16() else {
      fail("diagnostic requires the selected BF16 sm121 device")
    }
    let budget = @runtime.Budget::for_text(
      model.program,
      weight_bytes=model.weights.byte_count(),
    )
    guard budget.total_allocation_bytes <= 100000000000L &&
      budget.total_allocation_bytes <= capability.total_memory_bytes() else {
      fail("diagnostic allocation exceeds the selected device budget")
    }
    let weights = @weights.Loaded::new(model.weights)
    group.add_defer(() => weights.close())
    weights.prepare(
      context,
      root,
      maximum_device_bytes=budget.weight_bytes,
      scratch=FixedArray::make(4194304, b'\x00'),
      limits=@bootstrap.streaming_limits(),
    )
    let modules = @runtime.ModuleSet::new()
    group.add_defer(() => modules.close())
    modules.prepare(context, images)
    let stream = context.create_stream()
    group.add_defer(() => stream.close())
    let storage = @text.Storage::new(model.program)
    group.add_defer(() => storage.close())
    storage.prepare(context, maximum_device_bytes=budget.storage_bytes)
    let frame = @text.Frame::new(model.program)
    group.add_defer(() => frame.close())
    frame.prepare(
      context,
      modules=modules.bindings(),
      metadata=storage.metadata(),
      layers=weights.layer_weights(),
      weights=weights.text_weights(),
      states=storage.states(),
      buffers=storage.buffers(),
      sampled_tokens=storage.sampled_tokens(),
    )
    let runner = @request.Runner::new(model.program)
    group.add_defer(() => runner.close())
    runner.prepare(context, stream, frame, storage)
    let stops = ReadOnlyArray::from_array(@prompt.stop_token_ids())
    let rows : Array[Json] = []
    for index, request in requests {
      runner.begin(
        request.prompt,
        maximum_new_tokens=request.maximum_new_tokens,
        stop_token_ids=stops,
        context_ceiling=request.context_ceiling,
      )
      let files : Array[String] = []
      let mut observed = 0
      while runner.submit_next() {
        runner.wait()
        if runner.sampled_tokens() > observed {
          let path = "request-\{index}-step-\{observed}.f32"
          let logits = frame.logits()
          @fs.write_file(
            output + "/" + path,
            logits.copy_to_host(0L, logits.byte_count()),
            create_mode=CreateNew,
          )
          files.push(path)
          observed = runner.sampled_tokens()
        }
      }
      let tokens = runner.emitted_tokens().to_owned()
      let reason = match runner.finish_reason() {
        Some(StopSequence) => "stop"
        Some(TokenLimit) => "length"
        Some(Cancelled) => "cancelled"
        None => "incomplete"
      }
      let row : Json = {
        "id": Json(request.id),
        "token_ids": Json(tokens),
        "finish_reason": Json(reason),
        "gpu_execution": true,
      }
      @fs.write_file(
        output + "/result-\{index}.json",
        row.stringify(indent=2),
        create_mode=CreateNew,
      )
      rows.push({ "id": Json(request.id), "logits_files": Json(files) })
    }
    let report : Json = {
      "format": "lunaflux.minicpm-v47-native-logits.v1",
      "requests": Json(rows),
      "gpu_execution": true,
      "production_timing": false,
    }
    @fs.write_file(
      output + "/logits.json",
      report.stringify(indent=2),
      create_mode=CreateNew,
    )
  }
  println("native MiniCPM diagnostic complete: " + output)
}

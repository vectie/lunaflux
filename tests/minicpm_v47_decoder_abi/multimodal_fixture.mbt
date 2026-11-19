///|
fn run_multimodal_fixture(
  spec : @model.Spec,
  program : @text.Program,
  context : @device.Context,
  text_frame : @text.Frame,
  text_storage : @text.Storage,
) -> Unit raise {
  let initial = live_resources()
  let shared = context.allocate(158662656L)
  defer shared.close()
  let vision_program = @vision.Program::new(
    spec,
    maximum_patches=1024,
    maximum_crop_patches=1024,
  )
  let weights = @vision.Weights::borrow(spec, vision_program, context, fn(_) {
    return shared
  })
  let encoder = @vision.Execution::new(vision_program)
  defer encoder.close()
  encoder.prepare(
    context,
    b"fake-vision-module",
    weights,
    maximum_mutable_bytes=vision_program.storage_bytes(),
  )
  let bridge_program = @multimodal.Program::new(program)
  let unavailable = context.load_module(b"closed-scatter-module")
  unavailable.close()
  let incomplete = @multimodal.Frame::new(bridge_program)
  let before_incomplete = live_resources()
  assert_true(
    try
      incomplete.prepare(
        context,
        aot=unavailable,
        text_frame~,
        text_storage~,
        features=encoder.feature_buffer(),
        maximum_metadata_bytes=bridge_program.metadata_bytes(),
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_eq(live_resources() - before_incomplete, 2)
  assert_true(
    try incomplete.launches(Prefill) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  incomplete.close()
  incomplete.close()
  assert_eq(live_resources(), before_incomplete)
  let scatter_module = context.load_module(b"fake-scatter-module")
  defer scatter_module.close()
  let bridge = @multimodal.Frame::new(bridge_program)
  defer bridge.close()
  let created = creates()
  assert_true(
    try
      bridge.prepare(
        context,
        aot=scatter_module,
        text_frame~,
        text_storage~,
        features=encoder.feature_buffer(),
        maximum_metadata_bytes=bridge_program.metadata_bytes() - 1L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_eq(creates(), created)
  bridge.prepare(
    context,
    aot=scatter_module,
    text_frame~,
    text_storage~,
    features=encoder.feature_buffer(),
    maximum_metadata_bytes=bridge_program.metadata_bytes(),
  )
  assert_eq(bridge.launches(Prefill).length(), 925)
  assert_eq(bridge.launches(Decode).length(), 924)
  let stream = context.create_stream()
  defer stream.close()
  let runner = @request.Runner::new(program)
  defer runner.close()
  runner.prepare_launches(
    context,
    stream,
    text_storage,
    prefill=bridge.launches(Prefill),
    decode=bridge.launches(Decode),
    policy=OrderedKernelEagerOnly,
  )
  let plan = @processor.Plan::build(spec, height=56, width=56)
  let pixels = plan.prepare_rgb(
    Bytes::from_array(Array::make(56 * 56 * 3, b'\x80'))[:],
    max_bytes=10000000L,
  )
  let layout = @vision_layout.Layout::build(
    spec,
    plan.target_sizes()[:],
    maximum_patches=1024,
    maximum_crop_patches=1024,
  )
  encoder.submit(@vision.Input::new(vision_program, layout, pixels))
  encoder.wait()
  let prompt = @multimodal.prepare_image_tokens(
    spec,
    ([100] + plan.placeholder().tokens()[:].to_owned())[:],
    [plan],
    maximum_tokens=4096,
  )
  let features = encoder.features()
  let wrong = @multimodal.prepare_image_tokens(
    spec,
    (plan.placeholder().tokens()[:].to_owned() +
    [100] +
    plan.placeholder().tokens()[:].to_owned())[:],
    [plan, plan],
    maximum_tokens=4096,
  )
  let before_mapping = (copies(), reads())
  assert_true(
    try bridge.bind_prompt(context, wrong, features) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_eq((copies(), reads()), before_mapping)
  let other_context = @device.open_context(0)
  assert_true(
    try bridge.bind_prompt(other_context, prompt, features) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  other_context.close()
  assert_true(
    try bridge.launches(Prefill) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  bridge.bind_prompt(context, prompt, features)
  assert_eq(copies() - before_mapping.0, 2UL)
  assert_eq(reads(), before_mapping.1)
  runner.begin(prompt.prompt(), maximum_new_tokens=2, stop_token_ids=[])
  let before = launches()
  measured_request_step(runner, produces_token=false)
  measured_request_step(runner, produces_token=false)
  set_sample_token(100)
  measured_request_step(runner, produces_token=true)
  set_sample_token(101)
  measured_request_step(runner, produces_token=true, poll=true)
  assert_eq(launches() - before, 3699UL)
  assert_eq(runner.emitted_tokens(), [100, 101])
  assert_true(runner.is_finished())
  // A later text request must disable the prior image map before prefill.
  let before_clear = copies()
  bridge.clear_prompt(context)
  assert_eq(copies() - before_clear, 1UL)
  let plain = @prompt.prepare_token_prompt(
    spec,
    Array::make(33, 1000)[:],
    maximum_tokens=4096,
  )
  runner.begin(plain, maximum_new_tokens=1, stop_token_ids=[])
  measured_request_step(runner, produces_token=false)
  measured_request_step(runner, produces_token=true)
  bridge.bind_prompt(context, prompt, features)
  runner.begin(prompt.prompt(), maximum_new_tokens=1, stop_token_ids=[])
  assert_true(runner.submit_next())
  runner.cancel()
  runner.close()
  stream.close()
  bridge.close()
  scatter_module.close()
  encoder.close()
  shared.close()
  assert_eq(live_resources(), initial)
  println(
    "MiniCPM-V 4.7 multimodal ABI:vision features -> two mapping uploads ->925-launch chunked prefill/924-launch decode; zero-allocation request steps, cancellation and text reuse passed; scripted driver only",
  )
}

///|
extern "C" fn copied_bytes() -> UInt64 = "lunaflux_device_worker_fake_copied_bytes"

///|
fn vision_frame_probe(
  spec : @model.Spec,
  program : @vision.Program,
  context : @device.Context,
  shared : @device.Allocation,
  weights : @vision.Weights,
) -> Unit raise {
  let aot = context.load_module(b"fake-vision-module")
  defer aot.close()
  let stream = context.create_stream()
  defer stream.close()
  let storage = @vision.Storage::new(program)
  defer storage.close()
  storage.prepare(context, maximum_device_bytes=program.storage_bytes())
  let frame = @vision.Frame::new(program)
  defer frame.close()
  frame.prepare(context, aot~, weights~, buffers=storage.buffers())
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  defer queue.close()
  let baseline = launches()
  measured_submit(queue)
  measured_submit(queue, reset=true)
  assert_eq(launches() - baseline, program.stages().length().to_uint64() * 2UL)
  let first = program.weight_names()[0]
  let overlapping = storage.buffers().region(State).allocation
  let alias_weights = @vision.Weights::borrow(spec, program, context, name => {
    if name == first {
      overlapping
    } else {
      shared
    }
  })
  let invalid = @vision.Frame::new(program)
  defer invalid.close()
  let created = creates()
  assert_true(
    (try? invalid.prepare(
      context,
      aot~,
      weights=alias_weights,
      buffers=storage.buffers(),
    ))
    is Err(_),
  )
  assert_eq(creates(), created)
}

///|
async fn run_vision_fixture() -> Unit {
  @async.with_task_group() <| group => {
    let spec = @model.Spec::parse(
      @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
    )
    let program = @vision.Program::new(
      spec,
      maximum_patches=1024,
      maximum_crop_patches=1024,
    )
    let context = @device.open_context(0)
    group.add_defer(() => context.close())
    // Shared zero weight backing is an ABI-only fixture, not checkpoint values.
    let shared = context.allocate(158662656L)
    group.add_defer(() => shared.close())
    let weights = @vision.Weights::borrow(spec, program, context, _ => shared)
    let live = live_resources()
    vision_frame_probe(spec, program, context, shared, weights)
    assert_eq(live_resources(), live)
    let admission = @vision.Execution::new(program)
    let created = creates()
    assert_true(
      (try? admission.prepare(
        context,
        b"fake",
        weights,
        maximum_mutable_bytes=program.storage_bytes() - 1L,
      ))
      is Err(_),
    )
    assert_eq(creates(), created)
    admission.close()
    let closed = context.allocate(158662656L)
    let closed_weights = @vision.Weights::borrow(spec, program, context, fn(_) {
      return closed
    })
    closed.close()
    let partial = @vision.Execution::new(program)
    assert_true(
      (try? partial.prepare(
        context,
        b"fake",
        closed_weights,
        maximum_mutable_bytes=program.storage_bytes(),
      ))
      is Err(_),
    )
    assert_true(live_resources() > live)
    partial.close()
    partial.close()
    assert_eq(live_resources(), live)
    let pixel_plan = @processor.Plan::build(spec, height=56, width=56)
    let pixels = pixel_plan.prepare_rgb(
      Bytes::from_array(Array::make(56 * 56 * 3, b'\x80'))[:],
      max_bytes=10000000L,
    )
    let layout = @vision_layout.Layout::build(
      spec,
      pixel_plan.target_sizes()[:],
      maximum_patches=1024,
      maximum_crop_patches=1024,
    )
    let input = @vision.Input::new(program, layout, pixels)
    let execution = @vision.Execution::new(program)
    group.add_defer(() => execution.close())
    execution.prepare(
      context,
      b"fake-vision-module",
      weights,
      maximum_mutable_bytes=program.storage_bytes(),
    )
    assert_true(execution.is_ready())
    assert_true((try? execution.features()) is Err(_))
    let before = (launches(), creates(), closes(), copies(), reads())
    execution.submit(input)
    assert_true(!execution.is_ready())
    assert_true((try? execution.submit(input)) is Err(_))
    execution.wait()
    assert_eq(execution.features().rows(), 64)
    assert_eq(execution.features().allocation().byte_count(), 64L * 2048L * 2L)
    assert_eq(launches() - before.0, 318UL)
    assert_eq((creates(), closes(), reads()), (before.1, before.2, before.4))
    assert_eq(copies() - before.3, 13UL)
    execution.submit(input)
    assert_true(execution.poll())
    assert_true(execution.is_ready())
    execution.submit(input)
    execution.cancel()
    assert_true(execution.is_ready())
    assert_true((try? execution.features()) is Err(_))
    set_fault(1)
    assert_true((try? execution.submit(input)) is Err(_))
    assert_true(!execution.is_ready())
    execution.cancel()
    execution.submit(input)
    execution.wait()
    assert_eq(execution.features().rows(), 64)
    execution.submit(input)
    execution.close()
    execution.close()
    assert_eq(live_resources(), live)
    assert_true((try? execution.features()) is Err(_))
    let config = @utf8.decode(
      @fs.read_file("model/minicpm_v47/config.fixture.json").binary(),
    )
    let four_spec = @model.Spec::parse(
      @utf8.encode(config.replace(old="\"16x\"", new="\"4x\""))[:],
    )
    let four_program = @vision.Program::new(
      four_spec,
      maximum_patches=1024,
      maximum_crop_patches=1024,
    )
    let four_weights = @vision.Weights::borrow(
      four_spec,
      four_program,
      context,
      _ => shared,
    )
    vision_frame_probe(four_spec, four_program, context, shared, four_weights)
    let four_plan = @processor.Plan::build(four_spec, height=56, width=56)
    let four_pixels = four_plan.prepare_rgb(
      Bytes::from_array(Array::make(56 * 56 * 3, b'\x80'))[:],
      max_bytes=10000000L,
    )
    let four_layout = @vision_layout.Layout::build(
      four_spec,
      four_plan.target_sizes()[:],
      maximum_patches=1024,
      maximum_crop_patches=1024,
    )
    let four_input = @vision.Input::new(four_program, four_layout, four_pixels)
    let four_execution = @vision.Execution::new(four_program)
    defer four_execution.close()
    four_execution.prepare(
      context,
      b"fake-vision-module",
      four_weights,
      maximum_mutable_bytes=four_program.storage_bytes(),
    )
    let before_four = (launches(), copies(), reads())
    four_execution.submit(four_input)
    four_execution.wait()
    assert_eq(launches() - before_four.0, 305UL)
    assert_eq(copies() - before_four.1, 10UL)
    assert_eq(reads(), before_four.2)
    assert_eq(four_execution.features().rows(), 256)
    four_execution.close()
    assert_eq(live_resources(), live)
    @checkpoint_fixture.with_checkpoint((model, checkpoint, root) => {
      let plan = @vision.WeightPlan::new(model, program, checkpoint)
      let loaded = @vision.Loaded::new(plan)
      defer loaded.close()
      let before_budget = creates()
      assert_true(
        (try? loaded.prepare(
          context,
          root,
          maximum_device_bytes=plan.byte_count() - 1L,
          scratch=FixedArray::make(3, b'\x00'),
          limits=@checkpoint_fixture.limits(),
        ))
        is Err(_),
      )
      assert_eq(creates(), before_budget)
      let before_upload = copies()
      let before_bytes = copied_bytes()
      loaded.prepare(
        context,
        root,
        maximum_device_bytes=plan.byte_count(),
        scratch=FixedArray::make(4194304, b'\x00'),
        limits=@checkpoint_fixture.limits(),
      )
      assert_eq(creates() - before_budget, 459UL)
      assert_true(copies() > before_upload)
      assert_eq(copied_bytes() - before_bytes, plan.byte_count().to_uint64())
      vision_frame_probe(model, program, context, shared, loaded.weights())
      loaded.close()
      assert_eq(live_resources(), live)
      let incomplete = @vision.Loaded::new(plan)
      defer incomplete.close()
      let before_partial = creates()
      // Missing startup authority after planning fails transfer, after final
      // allocations have been tracked. It must release all459 without uploads.
      root.close()
      assert_true(
        (try? incomplete.prepare(
          context,
          root,
          maximum_device_bytes=plan.byte_count(),
          scratch=FixedArray::make(3, b'\x00'),
          limits=@checkpoint_fixture.limits(),
        ))
        is Err(_),
      )
      assert_eq(creates() - before_partial, 459UL)
      assert_true((try? incomplete.weights()) is Err(_))
      incomplete.close()
      incomplete.close()
      assert_eq(live_resources(), live)
    })
  }
  println(
    "MiniCPM-V 4.7 vision ABI:459-tensor sparse-zero upload,318-stage frame, zero-allocation queue replay,13 input copies/no readback, single-flight completion/cancel/reuse, partial startup and active close passed; no GPU computation",
  )
}

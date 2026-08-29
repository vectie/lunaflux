///|
fn probe_kernel_arguments(
  contract : @launch_contract.PagedV4AotLaunchContract,
  arena : @device.Allocation,
) -> @device.KernelArguments raise {
  let arguments : Array[@device.KernelArgument] = []
  for operand in contract.operands() {
    let offset = match operand.role() {
      RuntimeInput(StepCounts) => 0L
      ActivationInput(_) => 32L
      WeightInput(_) => 64L
      WeightScaleInput(_, _) => 80L
      ActivationOutput(_) => 96L
      _ => raise I8PhysicalProbeFailure("unexpected physical I8 operand role")
    }
    arguments.push(
      @device.KernelArgument::device_region(
        arena,
        offset~,
        byte_count=operand.byte_count(),
        alignment=operand.alignment(),
      ),
    )
  }
  @device.KernelArguments::new(arguments)
}

///|
fn upload_i8_probe_fixture(
  context : @device.Context,
  arena : @device.Allocation,
) -> Unit raise {
  let counts = probe_counts_bytes()
  let input = probe_input_bytes()
  let weight = probe_weight_bytes()
  let scale = probe_scale_bytes()
  let output = FixedArray::make(24, b'\x00')
  context.copy_from_fixed_host(arena, counts, 0L, 0L, 20L)
  context.copy_from_fixed_host(arena, input, 0L, 32L, 24L)
  context.copy_from_fixed_host(arena, weight, 0L, 64L, 16L)
  context.copy_from_fixed_host(arena, scale, 0L, 80L, 16L)
  context.copy_from_fixed_host(arena, output, 0L, 96L, 24L)
}

///|
fn probe_device_dimensions(
  dimensions : @launch_contract.AotLaunchDimensions,
) -> @device.KernelLaunchDimensions raise {
  @device.KernelLaunchDimensions::new(
    grid_x=dimensions.grid_x(),
    grid_y=dimensions.grid_y(),
    grid_z=dimensions.grid_z(),
    block_x=dimensions.block_x(),
    block_y=dimensions.block_y(),
    block_z=dimensions.block_z(),
    shared_memory_bytes=dimensions.shared_memory_bytes(),
  )
}

///|
fn run_i8_probe(admission : I8ProbeAdmission, cycles : Int) -> Double raise {
  let context = @device.open_context(0)
  defer (try! context.close())
  require_probe(
    context.capability().compute_major() ==
    admission.device_numeric.compute_major() &&
    context.capability().compute_minor() ==
    admission.device_numeric.compute_minor(),
    "opened CUDA context does not match admitted I8 device",
  )
  let stream = context.create_stream()
  defer (try! stream.close())
  let arena = context.allocate(128L)
  defer (try! arena.close())
  upload_i8_probe_fixture(context, arena)
  let family = admission.selected.entry_point().family()
  guard admission.artifacts.find_module(family.module_digest())
    is Some(artifact_module) &&
    admission.artifacts.entry_point(admission.selected.entry_point())
    is Some(artifact_entry) else {
    raise I8PhysicalProbeFailure("admitted I8 artifact cannot be resolved")
  }
  let loaded_module = context.load_module(artifact_module.module_bytes())
  defer (try! loaded_module.close())
  let function = loaded_module.load_function(
    @utf8.encode(artifact_entry.function_symbol()),
  )
  defer (try! function.close())
  let launch = @device.OrderedKernelLaunch::new(
    function,
    probe_device_dimensions(admission.selected.dimensions()),
    probe_kernel_arguments(admission.selected, arena),
  )
  let executor = context.create_ordered_kernel_executor(
    stream,
    [launch],
    policy=OrderedKernelCaptureRequired,
  )
  defer (try! executor.close())
  require_probe(
    executor.execution_mode() == OrderedKernelCaptured,
    "capture-required I8 executor did not capture",
  )
  let mut maximum_error = 0.0
  for _cycle in 0..<cycles {
    executor.launch_captured()
    executor.record_completion()
    executor.wait_completion()
    maximum_error = maximum_error.max(
      validate_i8_projection_output(arena.copy_to_host(96L, 24L)),
    )
    executor.reset()
  }
  maximum_error
}

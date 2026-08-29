///|
let buffer_sizes : FixedArray[Int] = [
  20, 12, 128, 24, 8, 24, 32, 16, 16, 48, 12, 48, 12, 8, 12, 12, 24, 1024, 1024,
  32, 24, 24, 8, 24, 64, 64, 64, 24, 24, 8, 24, 128, 96,
]

///|
let fixture_kinds : FixedArray[Int] = [
  0, 1, 2, 21, 3, 21, 4, 5, 6, 21, 7, 21, 8, 9, 10, 11, 21, 12, 13, 14, 21, 21, 15,
  21, 16, 17, 18, 21, 21, 19, 21, 20, 21,
]

///|
let buffer_alignments : FixedArray[Int64] = [
  4L, 4L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 4L, 2L, 4L, 4L, 4L, 4L, 2L, 2L, 2L, 2L,
  2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L,
]

///|
let argument_buffer_indices : FixedArray[Int] = [
  0, 1, 2, 3, 0, 3, 4, 5, 0, 5, 6, 7, 8, 9, 0, 10, 9, 11, 0, 10, 12, 13, 14, 15,
  11, 16, 17, 18, 0, 16, 19, 20, 0, 3, 20, 21, 0, 21, 22, 23, 0, 23, 24, 25, 26,
  27, 0, 21, 27, 28, 0, 28, 29, 30, 0, 30, 31, 32,
]

///|
let graph_dimensions : FixedArray[Int] = [
  1, 1, 1, 256, 1, 1, 0, 3, 1, 1, 256, 1, 1, 1024, 1, 1, 1, 256, 1, 1, 0, 1, 1, 1,
  256, 1, 1, 0, 3, 2, 1, 1, 1, 1, 0, 1, 1, 1, 256, 1, 1, 0, 1, 1, 1, 256, 1, 1, 0,
  3, 1, 1, 256, 1, 1, 1024, 1, 1, 1, 256, 1, 1, 0, 1, 1, 1, 256, 1, 1, 0, 3, 1, 1,
  256, 1, 1, 1024, 1, 1, 1, 256, 1, 1, 0,
]

///|
fn graph_readbacks(
  arena : @device.Allocation,
  buffer_offsets : FixedArray[Int64],
) -> FixedArray[Bytes] raise @device.DeviceError {
  let indices : FixedArray[Int] = [
    3, 5, 9, 11, 16, 17, 18, 20, 21, 23, 27, 28, 30, 32,
  ]
  FixedArray::makei(indices.length(), index => {
    let source = indices[index]
    arena.copy_to_host(buffer_offsets[source], buffer_sizes[source].to_int64())
  })
}

///|
async fn run_captured_graph(
  module_paths : FixedArray[String],
  cycles : Int,
) -> (Int, Double) {
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let buffer_offsets = FixedArray::make(buffer_sizes.length(), 0L)
  let mut arena_bytes = 0L
  for index in 0..<buffer_sizes.length() {
    let alignment = buffer_alignments[index]
    arena_bytes = (arena_bytes + alignment - 1L) & -alignment
    buffer_offsets[index] = arena_bytes
    arena_bytes += buffer_sizes[index].to_int64()
  }
  let arena = context.allocate(arena_bytes)
  defer (try! arena.close())
  for index in 0..<buffer_sizes.length() {
    upload_fixture(
      context,
      arena,
      fixture_kinds[index],
      buffer_sizes[index],
      buffer_offsets[index],
    )
  }
  let images : Array[Bytes] = []
  for index in 0..<module_paths.length() {
    images.push(@fs.read_file(module_paths[index]).binary())
  }
  let pointwise_module = context.load_module(images[0])
  defer (try! pointwise_module.close())
  let qkv_module = context.load_module(images[1])
  defer (try! qkv_module.close())
  let dense_module = context.load_module(images[2])
  defer (try! dense_module.close())
  let mlp_module = context.load_module(images[3])
  defer (try! mlp_module.close())
  let lm_head_module = context.load_module(images[4])
  defer (try! lm_head_module.close())
  let attention_module = context.load_module(images[5])
  defer (try! attention_module.close())
  let embedding = pointwise_module.load_function(b"lunaflux_probe_embedding_h4")
  defer (try! embedding.close())
  let rms = pointwise_module.load_function(b"lunaflux_probe_rms_norm_h4")
  defer (try! rms.close())
  let residual = pointwise_module.load_function(b"lunaflux_probe_residual_h4")
  defer (try! residual.close())
  let rope = pointwise_module.load_function(b"lunaflux_probe_rope_qkv_h4")
  defer (try! rope.close())
  let qkv = qkv_module.load_function(
    b"lunaflux_luna_qkv_projection_bf16_reference_v1_op_1_ep_2",
  )
  defer (try! qkv.close())
  let dense = dense_module.load_function(
    b"lunaflux_luna_dense_projection_bf16_reference_v1_op_4_ep_5",
  )
  defer (try! dense.close())
  let mlp = mlp_module.load_function(
    b"lunaflux_luna_gated_mlp_bf16_reference_v1_op_5_ep_6",
  )
  defer (try! mlp.close())
  let lm_head = lm_head_module.load_function(
    b"lunaflux_luna_language_model_head_bf16_reference_v1_op_6_ep_7",
  )
  defer (try! lm_head.close())
  let attention = attention_module.load_function(
    b"lunaflux_paged_attention_bf16_reference_v1_ep_3",
  )
  defer (try! attention.close())
  let functions : FixedArray[@device.Function] = [
    embedding, rms, qkv, rope, attention, dense, residual, rms, mlp, residual, rms,
    lm_head,
  ]
  let starts : FixedArray[Int] = [
    0, 4, 8, 14, 18, 28, 32, 36, 40, 46, 50, 54, 58,
  ]
  let launches : Array[@device.OrderedKernelLaunch] = []
  for kernel in 0..<functions.length() {
    let arguments : Array[@device.KernelArgument] = []
    for argument in starts[kernel]..<starts[kernel + 1] {
      let buffer = argument_buffer_indices[argument]
      arguments.push(
        @device.KernelArgument::device_region(
          arena,
          offset=buffer_offsets[buffer],
          byte_count=buffer_sizes[buffer].to_int64(),
          alignment=buffer_alignments[buffer],
        ),
      )
    }
    let dimension = kernel * 7
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[kernel],
        @device.KernelLaunchDimensions::new(
          grid_x=graph_dimensions[dimension],
          grid_y=graph_dimensions[dimension + 1],
          grid_z=graph_dimensions[dimension + 2],
          block_x=graph_dimensions[dimension + 3],
          block_y=graph_dimensions[dimension + 4],
          block_z=graph_dimensions[dimension + 5],
          shared_memory_bytes=graph_dimensions[dimension + 6],
        ),
        @device.KernelArguments::new(arguments[:]),
      ),
    )
  }
  let executor = context.create_ordered_kernel_executor(
    stream,
    launches[:],
    policy=OrderedKernelCaptureRequired,
  )
  defer (try! executor.close())
  require_graph(
    executor.execution_mode() == OrderedKernelCaptured,
    "capture-required executor did not select captured mode",
  )
  let mut selected_token = -1
  let mut maximum_error = 0.0
  for cycle in 0..<cycles {
    executor.launch_captured()
    executor.record_completion()
    executor.wait_completion()
    let (token, error) = validate_graph_outputs(
      graph_readbacks(arena, buffer_offsets),
    )
    if cycle == 0 {
      selected_token = token
    } else {
      require_graph(
        token == selected_token,
        "host sampling changed across cycles",
      )
    }
    if error > maximum_error {
      maximum_error = error
    }
    executor.reset()
  }
  (selected_token, maximum_error)
}

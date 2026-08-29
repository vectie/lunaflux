///|
const SHAPE_MATRIX_CONTRACT_SHA256 : String = "a93252e0181653e1b90d535f1c28a79b065dd5cdeecc638bbddea94f927eb937"

///|
async fn matrix_authenticate_contract(path : String) -> Unit {
  let bytes = @fs.read_file(path).binary()
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  require_matrix(
    digest == SHAPE_MATRIX_CONTRACT_SHA256,
    "shape-matrix contract digest mismatch",
  )
}

///|
async fn matrix_authenticated_module(path : String) -> Bytes {
  let bytes = @fs.read_file(path).binary()
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  require_matrix(
    path.has_suffix("/sha256/" + digest + ".cubin"),
    "module path is not content-addressed by its exact bytes",
  )
  bytes
}

///|
fn matrix_upload_case(
  context : @device.Context,
  arena : @device.Allocation,
  offsets : FixedArray[Int64],
  case : ShapeMatrixCase,
) -> Unit raise {
  for index in 0..<matrix_buffer_sizes.length() {
    let bytes = matrix_case_buffer(index, case)
    context.copy_from_fixed_host(
      arena,
      bytes,
      0L,
      offsets[index],
      matrix_buffer_sizes[index].to_int64(),
    )
  }
}

///|
fn matrix_readbacks(
  arena : @device.Allocation,
  offsets : FixedArray[Int64],
) -> FixedArray[Bytes] raise @device.DeviceError {
  let indices : FixedArray[Int] = [
    3, 5, 9, 11, 16, 17, 18, 20, 21, 23, 27, 28, 30, 32,
  ]
  FixedArray::makei(indices.length(), index => {
    let source = indices[index]
    arena.copy_to_host(offsets[source], matrix_buffer_sizes[source].to_int64())
  })
}

///|
async fn run_shape_matrix(
  contract_path : String,
  module_paths : FixedArray[String],
  cycles : Int,
) -> (Int, Double) {
  matrix_authenticate_contract(contract_path)
  let images : Array[Bytes] = []
  for path in module_paths {
    images.push(matrix_authenticated_module(path))
  }
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let offsets = FixedArray::make(matrix_buffer_sizes.length(), 0L)
  let mut arena_bytes = 0L
  for index in 0..<matrix_buffer_sizes.length() {
    let alignment = matrix_buffer_alignments[index]
    arena_bytes = (arena_bytes + alignment - 1L) & -alignment
    offsets[index] = arena_bytes
    arena_bytes += matrix_buffer_sizes[index].to_int64()
  }
  let arena = context.allocate(arena_bytes)
  defer (try! arena.close())
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
      let buffer = matrix_argument_buffer_indices[argument]
      arguments.push(
        @device.KernelArgument::device_region(
          arena,
          offset=offsets[buffer],
          byte_count=matrix_buffer_sizes[buffer].to_int64(),
          alignment=matrix_buffer_alignments[buffer],
        ),
      )
    }
    let dimension = kernel * 7
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[kernel],
        @device.KernelLaunchDimensions::new(
          grid_x=matrix_graph_dimensions[dimension],
          grid_y=matrix_graph_dimensions[dimension + 1],
          grid_z=matrix_graph_dimensions[dimension + 2],
          block_x=matrix_graph_dimensions[dimension + 3],
          block_y=matrix_graph_dimensions[dimension + 4],
          block_z=matrix_graph_dimensions[dimension + 5],
          shared_memory_bytes=matrix_graph_dimensions[dimension + 6],
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
  require_matrix(
    executor.execution_mode() == OrderedKernelCaptured,
    "capture-required matrix did not select captured mode",
  )
  let cases = shape_matrix_cases()
  let selected = FixedArray::make(cases.length(), -1)
  let mut maximum_error = 0.0
  for _cycle in 0..<cycles {
    for index, case in cases {
      matrix_upload_case(context, arena, offsets, case)
      executor.launch_captured()
      executor.record_completion()
      executor.wait_completion()
      let (token, error) = matrix_validate_reference(
        case,
        matrix_readbacks(arena, offsets),
      )
      if selected[index] < 0 {
        selected[index] = token
      } else {
        require_matrix(
          selected[index] == token,
          "sample drift for \{case.name}",
        )
      }
      if error > maximum_error {
        maximum_error = error
      }
      executor.reset()
    }
  }
  (cycles * cases.length(), maximum_error)
}

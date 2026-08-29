///|
fn run_luna_tile_elementwise_canary(
  planned : LunaTileCudaAotInput,
  left : FixedArray[Double],
  right : FixedArray[Double],
  destination : FixedArray[Double],
) -> Int raise LunaTileError {
  if left.length() != right.length() || left.length() != destination.length() {
    raise InvalidProgram(Elementwise)
  }
  let decoded = decode_luna_tile_cuda_aot_plan(planned)
  match decoded.semantic {
    LunaTileDecodedResidualAdd(width~, ..) => {
      if width != Int64::from_int(destination.length()) {
        raise InvalidProgram(Elementwise)
      }
      for index in 0..<destination.length() {
        destination[index] = left[index] + right[index]
      }
      1
    }
    LunaTileDecodedNeutral => raise InvalidProgram(SemanticContract)
  }
}

///|
test "decoded residual-add AOT plan agrees with independent scalar referee" {
  let ids = FixedArray::make(3, LunaTileTensorId(1))
  ids[0] = LunaTileTensorId(1)
  ids[1] = LunaTileTensorId(2)
  ids[2] = LunaTileTensorId(3)
  let tensors : Array[LunaTileTensorView] = []
  for id in ids {
    tensors.push({
      id,
      dtype: BFloat16,
      address_space: Register,
      shape: ReadOnlyArray::from_array([4L]),
      strides_bytes: ReadOnlyArray::from_array([2L]),
      offset_bytes: 0L,
      alignment_bytes: 4L,
    })
  }
  let program = admit_luna_tile_program(
    LunaTileProgramInput::residual_add(
      left=ids[0],
      right=ids[1],
      output=ids[2],
      width=4L,
      loop_extents=[1L],
      tensors~,
      instructions=[
        LunaTileInstruction::elementwise(Add, ids[2], [ids[0], ids[1]]),
      ],
      constraints=[],
    ),
    LunaTileLimits::new(
      max_tensors=3,
      max_instructions=1,
      max_rank=1,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=4096,
      max_shared_memory_bytes=0L,
    ),
  )
  let left = FixedArray::from_array([-3.0, -0.0, 0.5, 1000.0])
  let right = FixedArray::from_array([2.0, 0.0, -0.25, -999.0])
  let actual = FixedArray::make(4, 0.0)
  let expected = FixedArray::make(4, 0.0)
  for index in 0..<4 {
    expected[index] = left[index] + right[index]
  }
  let planned = plan_luna_tile_cuda_aot_input(
    program,
    LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=0L,
    ),
  )
  let canary = run_luna_tile_elementwise_canary(planned, left, right, actual)
  assert_eq(canary, 1)
  assert_eq(actual, expected)
}

///|
test "AOT plan decoder rejects altered canonical bytes" {
  let ids = FixedArray::from_array([
    LunaTileTensorId(1),
    LunaTileTensorId(2),
    LunaTileTensorId(3),
  ])
  let tensors : Array[LunaTileTensorView] = []
  for id in ids {
    tensors.push({
      id,
      dtype: BFloat16,
      address_space: Register,
      shape: ReadOnlyArray::from_array([2L]),
      strides_bytes: ReadOnlyArray::from_array([2L]),
      offset_bytes: 0L,
      alignment_bytes: 2L,
    })
  }
  let program = admit_luna_tile_program(
    LunaTileProgramInput::residual_add(
      left=ids[0],
      right=ids[1],
      output=ids[2],
      width=2L,
      loop_extents=[1L],
      tensors~,
      instructions=[
        LunaTileInstruction::elementwise(Add, ids[2], [ids[0], ids[1]]),
      ],
      constraints=[],
    ),
    LunaTileLimits::new(
      max_tensors=3,
      max_instructions=1,
      max_rank=1,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=4096,
      max_shared_memory_bytes=0L,
    ),
  )
  let planned = plan_luna_tile_cuda_aot_input(
    program,
    LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=0L,
    ),
  )
  let altered_bytes = planned.canonical.to_array()
  altered_bytes[0] = b'X'
  let altered : LunaTileCudaAotInput = {
    program_digest: planned.program_digest,
    memory_plan: planned.memory_plan,
    canonical: Bytes::from_array(altered_bytes),
    digest: planned.digest,
  }
  try decode_luna_tile_cuda_aot_plan(altered) catch {
    InvalidProgram(CanonicalEncoding) => ()
    _ => fail("unexpected AOT plan decode failure")
  } noraise {
    _ => fail("altered AOT plan was decoded")
  }
  let truncated_bytes = Bytes::makei(planned.canonical.length() - 1, index => {
    planned.canonical[index]
  })
  let truncated : LunaTileCudaAotInput = {
    program_digest: planned.program_digest,
    memory_plan: planned.memory_plan,
    canonical: truncated_bytes,
    digest: LunaTileCudaAotDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(truncated_bytes)),
    ),
  }
  try decode_luna_tile_cuda_aot_plan(truncated) catch {
    InvalidProgram(CanonicalEncoding) => ()
    _ => fail("unexpected truncated AOT plan decode failure")
  } noraise {
    _ => fail("truncated AOT plan was decoded")
  }
}

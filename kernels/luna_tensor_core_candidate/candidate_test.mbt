///|
test "exact sm120 BF16 WMMA candidate is deterministic and authority-free" {
  let program = tensor_core_test_program() catch {
    error => fail("program: \{Repr(error)}")
  }
  let serial = tensor_core_test_serial(program) catch {
    error => fail("serial: \{Repr(error)}")
  }
  let parallel = tensor_core_test_parallel(program, serial) catch {
    error => fail("parallel: \{Repr(error)}")
  }
  let first = @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
    program,
    serial,
    parallel,
    tensor_core_test_contract(),
    tensor_core_test_limits(),
  )
  let repeated = @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
    program,
    serial,
    parallel,
    tensor_core_test_contract(),
    tensor_core_test_limits(),
  )
  assert_eq(first, repeated)
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.serial_oracle_digest(), serial.digest())
  assert_eq(first.parallel_plan_digest(), parallel.digest())
  assert_eq(first.tile_count(), 64)
  assert_false(first.manifest_bindable())
  assert_false(first.promotion_authorized())
  let source = @utf8.decode(first.source_bytes())
  for
    required in [
      "#include <mma.h>", "__CUDA_ARCH__ != 1200", "nvcuda::wmma::matrix_a", "nvcuda::wmma::matrix_b",
      "__nv_bfloat16, nvcuda::wmma::row_major", "nvcuda::wmma::accumulator", "nvcuda::wmma::mma_sync",
      "nvcuda::wmma::store_matrix_sync", "blockDim.x != 32", "gridDim.x != 64", "__restrict__",
    ] {
    assert_true(source.contains(required))
  }
  assert_false(source.contains("asm("))
  assert_false(source.contains("mma.sync"))
  let canonical = @utf8.decode(first.canonical_bytes())
  for
    required in [
      "target=sm_120", "family=wmma-bf16-m16n16k16-f32", "shape=16,16,16", "left_layout=row_major",
      "right_layout=row_major", "accumulator_dtype=f32", "operand_alignment_bytes=32",
      "epilogue=identity", "sass_tensor_core_instruction_required=true", "sass_qualification_observed=false",
      "numeric_qualification_observed=false", "manifest_bindable=false", "promotion_authority=none",
    ] {
    assert_true(canonical.contains(required))
  }
}

///|
test "tensor-core contract rejects target shape layout dtype alignment and epilogue substitutions" {
  let cases : Array[() -> Unit raise] = [
    () => ignore(tensor_core_test_contract(compute_major=9, compute_minor=0)),
    () => ignore(tensor_core_test_contract(m=8)),
    () => ignore(tensor_core_test_contract(right_layout=TensorCoreColumnMajor)),
    () => {
      ignore(tensor_core_test_contract(accumulator=TensorCoreAccumulatorBf16))
    },
    () => ignore(tensor_core_test_contract(alignment=16)),
    () => ignore(tensor_core_test_contract(epilogue=TensorCoreBiasEpilogue)),
  ]
  for reject in cases {
    try reject() catch {
      @luna_tensor_core_candidate.LunaTileTensorCoreRejected(_) => ()
      _ => fail("unexpected tensor-core contract rejection")
    } noraise {
      _ => fail("unsupported tensor-core contract was accepted")
    }
  }
}

///|
test "tensor-core lowering rejects graph and parallel identity substitution" {
  let program = tensor_core_test_program()
  let serial = tensor_core_test_serial(program)
  let parallel = tensor_core_test_parallel(program, serial)
  let warp_program = tensor_core_test_program(barrier_scope=Warp)
  let warp_serial = tensor_core_test_serial(warp_program)
  let warp_parallel = tensor_core_test_parallel(warp_program, warp_serial)
  try
    @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
      warp_program,
      warp_serial,
      warp_parallel,
      tensor_core_test_contract(),
      tensor_core_test_limits(),
    )
  catch {
    @luna_tensor_core_candidate.LunaTileTensorCoreRejected(ProgramProof) => ()
    _ => fail("unexpected tensor-core graph rejection")
  } noraise {
    _ => fail("warp-barrier graph was accepted")
  }
  let misaligned_program = tensor_core_test_program(global_alignment=16L)
  let misaligned_serial = tensor_core_test_serial(misaligned_program)
  let misaligned_parallel = tensor_core_test_parallel(
    misaligned_program, misaligned_serial,
  )
  try
    @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
      misaligned_program,
      misaligned_serial,
      misaligned_parallel,
      tensor_core_test_contract(),
      tensor_core_test_limits(),
    )
  catch {
    @luna_tensor_core_candidate.LunaTileTensorCoreRejected(ProgramProof) => ()
    _ => fail("unexpected tensor-core alignment rejection")
  } noraise {
    _ => fail("16-byte global operand alignment was accepted")
  }
  let wrong_target_parallel = tensor_core_test_parallel(
    program,
    serial,
    compute_major=9,
  )
  try
    @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
      program,
      serial,
      wrong_target_parallel,
      tensor_core_test_contract(),
      tensor_core_test_limits(),
    )
  catch {
    @luna_tensor_core_candidate.LunaTileTensorCoreRejected(ParallelIdentity) =>
      ()
    _ => fail("unexpected parallel target rejection")
  } noraise {
    _ => fail("foreign-target parallel plan was accepted")
  }
  let foreign = tensor_core_test_program(tile_count=96L)
  let foreign_serial = tensor_core_test_serial(foreign)
  let foreign_parallel = tensor_core_test_parallel(foreign, foreign_serial)
  try
    @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
      program,
      serial,
      foreign_parallel,
      tensor_core_test_contract(),
      tensor_core_test_limits(),
    )
  catch {
    @luna_tensor_core_candidate.LunaTileTensorCoreRejected(ParallelIdentity) =>
      ()
    _ => fail("unexpected foreign parallel-plan rejection")
  } noraise {
    _ => fail("foreign parallel plan was accepted")
  }
  try
    @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
      program,
      serial,
      parallel,
      tensor_core_test_contract(),
      tensor_core_test_limits(source_bytes=32),
    )
  catch {
    @luna_tensor_core_candidate.LunaTileTensorCoreRejected(SourceLimit) => ()
    _ => fail("unexpected tensor-core source limit rejection")
  } noraise {
    _ => fail("unbounded tensor-core source was accepted")
  }
}

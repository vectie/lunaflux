///|
pub(all) enum LunaTileTensorCoreMatrixLayout {
  TensorCoreRowMajor
  TensorCoreColumnMajor
} derive(Debug, Eq)

///|
pub(all) enum LunaTileTensorCoreAccumulator {
  TensorCoreAccumulatorF32
  TensorCoreAccumulatorBf16
} derive(Debug, Eq)

///|
pub(all) enum LunaTileTensorCoreEpilogue {
  TensorCoreIdentityEpilogue
  TensorCoreBiasEpilogue
  TensorCoreSiluEpilogue
} derive(Debug, Eq)

///|
/// Exact source-only contract for the initial one-warp WMMA candidate.
pub struct LunaTileBf16Mma16x16x16Contract {
  priv compute_major : Int
  priv compute_minor : Int
  priv m : Int
  priv n : Int
  priv k : Int
  priv left_layout : LunaTileTensorCoreMatrixLayout
  priv right_layout : LunaTileTensorCoreMatrixLayout
  priv accumulator : LunaTileTensorCoreAccumulator
  priv output_layout : LunaTileTensorCoreMatrixLayout
  priv left_alignment_bytes : Int
  priv right_alignment_bytes : Int
  priv accumulator_alignment_bytes : Int
  priv output_alignment_bytes : Int
  priv epilogue : LunaTileTensorCoreEpilogue
  priv block_threads : Int
  priv maximum_registers_per_thread : Int
} derive(Debug, Eq)

///|
pub fn LunaTileBf16Mma16x16x16Contract::new(
  compute_major~ : Int,
  compute_minor~ : Int,
  m~ : Int,
  n~ : Int,
  k~ : Int,
  left_layout~ : LunaTileTensorCoreMatrixLayout,
  right_layout~ : LunaTileTensorCoreMatrixLayout,
  accumulator~ : LunaTileTensorCoreAccumulator,
  output_layout~ : LunaTileTensorCoreMatrixLayout,
  left_alignment_bytes~ : Int,
  right_alignment_bytes~ : Int,
  accumulator_alignment_bytes~ : Int,
  output_alignment_bytes~ : Int,
  epilogue~ : LunaTileTensorCoreEpilogue,
  block_threads~ : Int,
  maximum_registers_per_thread~ : Int,
) -> LunaTileBf16Mma16x16x16Contract raise LunaTileTensorCoreCandidateError {
  if compute_major != 12 || compute_minor != 0 {
    raise LunaTileTensorCoreRejected(Target)
  }
  if m != 16 || n != 16 || k != 16 {
    raise LunaTileTensorCoreRejected(Shape)
  }
  if left_layout != TensorCoreRowMajor ||
    right_layout != TensorCoreRowMajor ||
    output_layout != TensorCoreRowMajor {
    raise LunaTileTensorCoreRejected(Layout)
  }
  if accumulator != TensorCoreAccumulatorF32 {
    raise LunaTileTensorCoreRejected(DType)
  }
  if left_alignment_bytes != 32 ||
    right_alignment_bytes != 32 ||
    accumulator_alignment_bytes != 32 ||
    output_alignment_bytes != 32 {
    raise LunaTileTensorCoreRejected(Alignment)
  }
  if epilogue != TensorCoreIdentityEpilogue {
    raise LunaTileTensorCoreRejected(Epilogue)
  }
  if block_threads != 32 || maximum_registers_per_thread != 128 {
    raise LunaTileTensorCoreRejected(Resource)
  }
  {
    compute_major,
    compute_minor,
    m,
    n,
    k,
    left_layout,
    right_layout,
    accumulator,
    output_layout,
    left_alignment_bytes,
    right_alignment_bytes,
    accumulator_alignment_bytes,
    output_alignment_bytes,
    epilogue,
    block_threads,
    maximum_registers_per_thread,
  }
}

///|
pub struct LunaTileTensorCoreCandidateLimits {
  priv maximum_source_bytes : Int
  priv maximum_canonical_bytes : Int
}

///|
pub fn LunaTileTensorCoreCandidateLimits::new(
  maximum_source_bytes~ : Int,
  maximum_canonical_bytes~ : Int,
) -> LunaTileTensorCoreCandidateLimits raise LunaTileTensorCoreCandidateError {
  if maximum_source_bytes <= 0 || maximum_canonical_bytes <= 0 {
    raise LunaTileTensorCoreRejected(Contract)
  }
  { maximum_source_bytes, maximum_canonical_bytes }
}

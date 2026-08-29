///|
fn tensor_core_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
/// Emits one exact-sm120, one-warp WMMA candidate while retaining the serial
/// oracle and already-admitted parallel SIMT identities. WMMA source presence
/// is not SASS, correctness, performance, manifest, or promotion evidence.
pub fn lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
  program : @luna_tile_ir.LunaTileProgram,
  serial_oracle : @luna_tile_ir.LunaTileCudaTranslationUnit,
  parallel_plan : @luna_tile_ir.LunaTileParallelCandidate,
  contract : LunaTileBf16Mma16x16x16Contract,
  limits : LunaTileTensorCoreCandidateLimits,
) -> LunaTileBf16Mma16x16x16Candidate raise LunaTileTensorCoreCandidateError {
  let proof = @luna_tile_ir.admit_luna_tile_bf16_mma16x16x16_view(program) catch {
    _ => raise LunaTileTensorCoreRejected(ProgramProof)
  }
  if proof.program_digest() != program.digest() ||
    serial_oracle.program_digest() != program.digest() {
    raise LunaTileTensorCoreRejected(OracleIdentity)
  }
  if parallel_plan.program_digest() != program.digest() ||
    parallel_plan.serial_oracle_digest() != serial_oracle.digest() ||
    parallel_plan.planning_digest() != serial_oracle.planning_digest() ||
    parallel_plan.compute_capability() !=
    (contract.compute_major, contract.compute_minor) ||
    parallel_plan.tensor_core_eligibility() != TensorCoreEligibleUnqualified ||
    parallel_plan.execution_path() != ParallelSimtCandidate ||
    parallel_plan.mapping().tile_shape() != (contract.m, contract.n, contract.k) ||
    parallel_plan.mapping().block_threads() != contract.block_threads {
    raise LunaTileTensorCoreRejected(ParallelIdentity)
  }
  let tile_count = proof.tile_count()
  let entry_point = "lunaflux_lunatile_wmma_bf16_m16n16k16_sm120_v1"
  let source = tensor_core_wmma_source(entry_point, tile_count)
  if source.length() > limits.maximum_source_bytes {
    raise LunaTileTensorCoreRejected(SourceLimit)
  }
  let source_digest = LunaTileTensorCoreSourceDigest(tensor_core_sha256(source))
  let canonical = @utf8.encode(
    "schema=lunaflux-lunatile-tensor-core-candidate.v1\n" +
    "program_sha256=" +
    program.digest().as_hex() +
    "\nserial_oracle_sha256=" +
    serial_oracle.digest().as_hex() +
    "\nparallel_plan_sha256=" +
    parallel_plan.digest().as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nentry_point=" +
    entry_point +
    "\ntarget=sm_120\nfamily=wmma-bf16-m16n16k16-f32\n" +
    "left_tensor_id=" +
    proof.left_global().as_int().to_string() +
    "\nright_tensor_id=" +
    proof.right_global().as_int().to_string() +
    "\naccumulator_tensor_id=" +
    proof.accumulator_global().as_int().to_string() +
    "\noutput_tensor_id=" +
    proof.output_global().as_int().to_string() +
    "\ntile_count=" +
    tile_count.to_string() +
    "\nshape=16,16,16\nleft_dtype=bf16\nright_dtype=bf16\n" +
    "accumulator_dtype=f32\noutput_dtype=f32\nleft_layout=row_major\n" +
    "right_layout=row_major\noutput_layout=row_major\noperand_alignment_bytes=32\n" +
    "grid=" +
    tile_count.to_string() +
    ",1,1\nblock=32,1,1\nshared_memory_bytes=0\n" +
    "maximum_registers_per_thread=128\nepilogue=identity\n" +
    "sass_tensor_core_instruction_required=true\n" +
    "sass_qualification_observed=false\nnumeric_qualification_observed=false\n" +
    "performance_qualification_observed=false\ncompiler_authority=none\n" +
    "device_authority=none\nruntime_authority=none\nmanifest_authority=none\n" +
    "manifest_bindable=false\npromotion_authority=none\n",
  )
  if canonical.length() > limits.maximum_canonical_bytes {
    raise LunaTileTensorCoreRejected(SourceLimit)
  }
  {
    program_digest: program.digest(),
    serial_oracle_digest: serial_oracle.digest(),
    parallel_plan_digest: parallel_plan.digest(),
    tile_count,
    entry_point,
    source,
    source_digest,
    canonical,
    digest: LunaTileTensorCoreCandidateDigest(tensor_core_sha256(canonical)),
  }
}

///|
fn luna_tile_tensor_core_policy_name(
  policy : LunaTileTensorCorePolicy,
) -> String {
  match policy {
    ObserveEligibilityUseParallelSimt => "observe_eligibility_use_parallel_simt"
    RequireExternallyQualifiedTensorCore =>
      "require_externally_qualified_tensor_core"
  }
}

///|
fn luna_tile_tensor_core_eligibility_name(
  eligibility : LunaTileTensorCoreEligibility,
) -> String {
  match eligibility {
    TensorCoreEligibleUnqualified => "eligible_unqualified"
    TensorCoreIneligibleTarget => "ineligible_target"
    TensorCoreIneligibleShape => "ineligible_shape"
    TensorCoreNoMma => "no_mma"
  }
}

///|
fn luna_tile_parallel_canonical(
  program : LunaTileProgram,
  planning : LunaTileCudaAotInput,
  serial_oracle : LunaTileCudaTranslationUnit,
  entry_point_id : LunaTileEntryPointId,
  entry_point : String,
  source : Bytes,
  source_digest : LunaTileCudaSourceDigest,
  mapping : LunaTileParallelMapping,
  shared : LunaTileParallelSharedPlan,
  tensor_core : LunaTileTensorCoreEligibility,
  policy : LunaTileParallelPolicy,
) -> Bytes {
  let output = StringBuilder()
  output.write_string(
    "schema=lunaflux.lunatile-parallel-candidate.v1\n" +
    "program_sha256=" +
    program.digest.as_hex() +
    "\nplanning_sha256=" +
    planning.digest.as_hex() +
    "\nserial_oracle_sha256=" +
    serial_oracle.digest().as_hex() +
    "\nserial_oracle_role=qualification_only\nentry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\nentry_point=" +
    entry_point +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nsource_bytes=" +
    source.length().to_string() +
    "\ntarget=sm_" +
    policy.compute_major.to_string() +
    policy.compute_minor.to_string() +
    "\nblock_threads=" +
    mapping.block_threads.to_string() +
    "\nblock_count=" +
    mapping.block_count.to_string() +
    "\nwarps_per_block=" +
    mapping.warps_per_block.to_string() +
    "\ntile_m=" +
    mapping.tile_m.to_string() +
    "\ntile_n=" +
    mapping.tile_n.to_string() +
    "\ntile_k=" +
    mapping.tile_k.to_string() +
    "\ntile_count=" +
    mapping.tile_count.to_string() +
    "\nrounds_per_worker=" +
    mapping.rounds_per_worker.to_string() +
    "\ntile_mapping=round_major:block,warp,lane\nwarp_width=32" +
    "\npipeline_stages=" +
    policy.pipeline_stages.to_string() +
    "\nvector_bytes=" +
    policy.vector_bytes.to_string() +
    "\nshared_arena_bytes=" +
    shared.arena_bytes.to_string() +
    "\ntotal_shared_memory_bytes=" +
    shared.total_shared_memory_bytes.to_string() +
    "\nshared_reuse_count=" +
    shared.reuse_count.to_string() +
    "\ntensor_core_policy=" +
    luna_tile_tensor_core_policy_name(policy.tensor_core_policy) +
    "\ntensor_core_eligibility=" +
    luna_tile_tensor_core_eligibility_name(tensor_core) +
    "\nexecution_path=parallel_simt_candidate\n",
  )
  for placement in shared.placements {
    output.write_string(
      "shared_placement=" +
      placement.tensor.as_int().to_string() +
      "," +
      placement.offset_bytes.to_string() +
      "," +
      placement.size_bytes.to_string() +
      "," +
      placement.first_instruction.to_string() +
      "," +
      placement.last_instruction.to_string() +
      "," +
      placement.reused_slot.to_string() +
      "\n",
    )
  }
  output.write_string(
    "compiler_authority=none\nruntime_authority=none\n" +
    "device_authority=none\nmanifest_authority=none\n" +
    "promotion_authority=none\nmanifest_bindable=false\n",
  )
  @utf8.encode(output.to_string())
}

///|
/// Produces an inert, deterministic parallel SIMT candidate above a bound
/// serial semantic oracle. Tensor-core eligibility is descriptive only; this
/// boundary cannot accept or manufacture physical qualification.
pub fn specialize_luna_tile_parallel_candidate(
  program : LunaTileProgram,
  planning : LunaTileCudaAotInput,
  serial_oracle : LunaTileCudaTranslationUnit,
  entry_point_id : LunaTileEntryPointId,
  policy : LunaTileParallelPolicy,
  limits : LunaTileCudaLoweringLimits,
) -> LunaTileParallelCandidate raise LunaTileError {
  if planning.program_digest != program.digest ||
    serial_oracle.program_digest() != program.digest ||
    serial_oracle.planning_digest() != planning.digest ||
    planning.memory_plan.vector_bytes != policy.vector_bytes {
    raise InvalidProgram(ParallelPolicy)
  }
  validate_luna_tile_parallel_program(program)
  let mapping = plan_luna_tile_parallel_mapping(program, policy)
  let shared = plan_luna_tile_parallel_shared(program, policy)
  let tensor_core = luna_tile_tensor_core_eligibility(program, policy)
  if policy.tensor_core_policy == RequireExternallyQualifiedTensorCore {
    raise InvalidProgram(TensorCoreQualification)
  }
  let placed_program = luna_tile_parallel_placed_program(program, shared)
  let entry_point = "lunaflux_lunatile_parallel_simt_v1_ep_" +
    entry_point_id.as_int().to_string()
  let source = luna_tile_parallel_cuda_source(
    placed_program, mapping, shared, policy, entry_point,
  )
  if source.length() > limits.max_source_bytes {
    raise InvalidProgram(CudaLowering)
  }
  let source_digest = LunaTileCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let canonical = luna_tile_parallel_canonical(
    program, planning, serial_oracle, entry_point_id, entry_point, source, source_digest,
    mapping, shared, tensor_core, policy,
  )
  let digest = LunaTileParallelDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  {
    serial_oracle_digest: serial_oracle.digest(),
    program_digest: program.digest,
    planning_digest: planning.digest,
    compute_major: policy.compute_major,
    compute_minor: policy.compute_minor,
    mapping,
    shared,
    tensor_core,
    path: ParallelSimtCandidate,
    entry_point,
    source,
    source_digest,
    canonical,
    digest,
  }
}

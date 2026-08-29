///|
/// Re-runs the deterministic typed lowering and joins it to an exact opaque
/// physical evidence value. This is the evidence-aware counterpart to the
/// deliberately rejecting evidence-free tensor-core policy. It retains both
/// serial and parallel fallback identities and grants no production authority.
pub fn lower_and_bind_externally_qualified_luna_tile_bf16_mma16x16x16(
  program : @luna_tile_ir.LunaTileProgram,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
  contract : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Contract,
  candidate_limits : @luna_tensor_core_candidate.LunaTileTensorCoreCandidateLimits,
  evidence : AdmittedLunaTensorCorePhysicalEvidence,
) -> QualifiedLunaTileBf16Mma16x16x16Artifact raise LunaTensorCorePhysicalEvidenceError {
  let candidate = @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
    program, serial, parallel, contract, candidate_limits,
  ) catch {
    _ => raise LunaTensorCorePhysicalEvidenceMismatch(Candidate)
  }
  if candidate.digest().as_hex() != evidence.candidate_digest ||
    candidate.source_digest().as_hex() != evidence.source_digest ||
    candidate.serial_oracle_digest() != evidence.serial_oracle_digest ||
    candidate.parallel_plan_digest() != evidence.parallel_fallback_digest ||
    serial.digest() != evidence.serial_oracle_digest ||
    parallel.digest() != evidence.parallel_fallback_digest {
    raise LunaTensorCorePhysicalEvidenceMismatch(Candidate)
  }
  { candidate, evidence }
}

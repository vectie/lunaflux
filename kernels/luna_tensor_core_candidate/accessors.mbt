///|
pub fn LunaTileBf16Mma16x16x16Candidate::program_digest(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> @luna_tile_ir.LunaTileProgramDigest {
  self.program_digest
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::serial_oracle_digest(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> @luna_tile_ir.LunaTileCudaTranslationDigest {
  self.serial_oracle_digest
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::parallel_plan_digest(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> @luna_tile_ir.LunaTileParallelDigest {
  self.parallel_plan_digest
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::tile_count(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> Int {
  self.tile_count
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::entry_point(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> String {
  self.entry_point
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::source_bytes(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> Bytes {
  self.source
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::source_digest(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> LunaTileTensorCoreSourceDigest {
  self.source_digest
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::canonical_bytes(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::digest(
  self : LunaTileBf16Mma16x16x16Candidate,
) -> LunaTileTensorCoreCandidateDigest {
  self.digest
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::manifest_bindable(
  _self : LunaTileBf16Mma16x16x16Candidate,
) -> Bool {
  false
}

///|
pub fn LunaTileBf16Mma16x16x16Candidate::promotion_authorized(
  _self : LunaTileBf16Mma16x16x16Candidate,
) -> Bool {
  false
}

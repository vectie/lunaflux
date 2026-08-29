///|
const CANDIDATE_SET_MAX_OPERATIONS : Int = 4096

///|
const CANDIDATE_SET_MAX_FILE_BYTES : Int = 16777216

///|
const CANDIDATE_SET_MAX_TOTAL_BYTES : Int = 1073741824

///|
pub struct LunaBf16CandidateSetLimits {
  priv max_operations : Int
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
  priv max_total_bytes : Int
}

///|
pub fn LunaBf16CandidateSetLimits::new(
  max_operations~ : Int,
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
  max_total_bytes~ : Int,
) -> LunaBf16CandidateSetLimits raise LunaBf16KernelProducerError {
  if max_operations < 9 ||
    max_operations > CANDIDATE_SET_MAX_OPERATIONS ||
    max_source_bytes <= 0 ||
    max_source_bytes > CANDIDATE_SET_MAX_FILE_BYTES ||
    max_recipe_bytes <= 0 ||
    max_recipe_bytes > CANDIDATE_SET_MAX_FILE_BYTES ||
    max_total_bytes <= 0 ||
    max_total_bytes > CANDIDATE_SET_MAX_TOTAL_BYTES {
    raise Invalid(Limits)
  }
  { max_operations, max_source_bytes, max_recipe_bytes, max_total_bytes }
}

///|
priv struct LunaBf16CandidatePayload {
  key : String
  family : String
  source : Bytes
  source_sha256 : String
  recipe : Bytes
  recipe_sha256 : String
}

///|
/// Opaque, root-free compiler input derived only from one authenticated model
/// plan and the three product-owned BF16 lowering families.
pub struct LunaBf16CandidateSet {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv candidates : ReadOnlyArray[LunaBf16KernelCandidate]
  priv payloads : ReadOnlyArray[LunaBf16CandidatePayload]
  priv declaration : Bytes
  priv declaration_sha256 : String
  priv inventory : Bytes
  priv inventory_sha256 : String
  priv total_bytes : Int
}

///|
pub fn LunaBf16CandidateSet::model_identity(
  self : LunaBf16CandidateSet,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaBf16CandidateSet::target(
  self : LunaBf16CandidateSet,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaBf16CandidateSet::profile(
  self : LunaBf16CandidateSet,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn LunaBf16CandidateSet::compiler_policy(
  self : LunaBf16CandidateSet,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn LunaBf16CandidateSet::candidates(
  self : LunaBf16CandidateSet,
) -> ReadOnlyArray[LunaBf16KernelCandidate] {
  self.candidates
}

///|
pub fn LunaBf16CandidateSet::operation_count(
  self : LunaBf16CandidateSet,
) -> Int {
  self.candidates.length()
}

///|
pub fn LunaBf16CandidateSet::candidate_key(
  self : LunaBf16CandidateSet,
  index : Int,
) -> String? {
  self.payloads.get(index).map(payload => payload.key)
}

///|
pub fn LunaBf16CandidateSet::candidate_source_bytes(
  self : LunaBf16CandidateSet,
  index : Int,
) -> Bytes? {
  self.payloads.get(index).map(payload => payload.source)
}

///|
pub fn LunaBf16CandidateSet::candidate_recipe_bytes(
  self : LunaBf16CandidateSet,
  index : Int,
) -> Bytes? {
  self.payloads.get(index).map(payload => payload.recipe)
}

///|
pub fn LunaBf16CandidateSet::candidate_set_bytes(
  self : LunaBf16CandidateSet,
) -> Bytes {
  self.declaration
}

///|
pub fn LunaBf16CandidateSet::candidate_set_sha256(
  self : LunaBf16CandidateSet,
) -> String {
  self.declaration_sha256
}

///|
pub fn LunaBf16CandidateSet::inventory_bytes(
  self : LunaBf16CandidateSet,
) -> Bytes {
  self.inventory
}

///|
pub fn LunaBf16CandidateSet::inventory_sha256(
  self : LunaBf16CandidateSet,
) -> String {
  self.inventory_sha256
}

///|
pub fn LunaBf16CandidateSet::total_bytes(self : LunaBf16CandidateSet) -> Int {
  self.total_bytes
}

///|
/// Exact maximum live-shape ABI understood by one catalog-v2 entry point.
///
/// Unlike a full-prefill profile, rows may contain different prefill/decode
/// lengths. The three capacities bind the packed Int32 metadata allocations;
/// live values remain bounded by v2 `StepCounts` at execution time.
pub struct PagedKernelProfile {
  id : KernelProfileId
  max_query_rows : Int
  max_query_tokens : Int
  max_page_table_entries : Int
} derive(Debug, Eq)

///|
pub fn PagedKernelProfile::new(
  id~ : KernelProfileId,
  max_query_rows~ : Int,
  max_query_tokens~ : Int,
  max_page_table_entries~ : Int,
) -> PagedKernelProfile raise LaunchContractError {
  if max_query_rows <= 0 || max_query_tokens <= 0 || max_page_table_entries <= 0 {
    raise InvalidProfile(NonPositiveDimension)
  }
  { id, max_query_rows, max_query_tokens, max_page_table_entries }
}

///|
pub fn PagedKernelProfile::id(self : PagedKernelProfile) -> KernelProfileId {
  self.id
}

///|
pub fn PagedKernelProfile::max_query_rows(self : PagedKernelProfile) -> Int {
  self.max_query_rows
}

///|
pub fn PagedKernelProfile::max_query_tokens(self : PagedKernelProfile) -> Int {
  self.max_query_tokens
}

///|
pub fn PagedKernelProfile::max_page_table_entries(
  self : PagedKernelProfile,
) -> Int {
  self.max_page_table_entries
}

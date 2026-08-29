///|
/// Runtime row kind for one measured paged-attention step. This is profiler
/// evidence vocabulary only; it grants no scheduler or execution authority.
pub(all) enum LunaPagedProfileRowPhase {
  PagedPrefill
  PagedDecode
} derive(Debug, Eq)

///|
fn LunaPagedProfileRowPhase::stable_int(self : LunaPagedProfileRowPhase) -> Int {
  match self {
    PagedPrefill => 1
    PagedDecode => 2
  }
}

///|
/// Exact geometry for one live row. `cache_position` is the first query-token
/// position, `context_length` is the visible context after the row, and the
/// block-table slice addresses `block_count` packed page identities starting
/// at `page_table_start`.
pub struct LunaPagedProfileRow {
  priv phase : LunaPagedProfileRowPhase
  priv query_length : Int
  priv context_length : Int
  priv cache_position : Int
  priv page_table_start : Int
  priv block_count : Int
} derive(Debug, Eq)

///|
pub fn LunaPagedProfileRow::new(
  phase~ : LunaPagedProfileRowPhase,
  query_length~ : Int,
  context_length~ : Int,
  cache_position~ : Int,
  page_table_start~ : Int,
  block_count~ : Int,
) -> LunaPagedProfileRow raise LunaKernelProfileError {
  if query_length <= 0 ||
    context_length <= 0 ||
    cache_position < 0 ||
    cache_position > 2_147_483_647 - query_length ||
    context_length != cache_position + query_length ||
    page_table_start < 0 ||
    block_count <= 0 {
    raise LunaKernelProfileInvalid(Row)
  }
  if phase == PagedDecode && query_length != 1 {
    raise LunaKernelProfileInvalid(Row)
  }
  {
    phase,
    query_length,
    context_length,
    cache_position,
    page_table_start,
    block_count,
  }
}

///|
pub fn LunaPagedProfileRow::phase(
  self : LunaPagedProfileRow,
) -> LunaPagedProfileRowPhase {
  self.phase
}

///|
pub fn LunaPagedProfileRow::query_length(self : LunaPagedProfileRow) -> Int {
  self.query_length
}

///|
pub fn LunaPagedProfileRow::context_length(self : LunaPagedProfileRow) -> Int {
  self.context_length
}

///|
pub fn LunaPagedProfileRow::cache_position(self : LunaPagedProfileRow) -> Int {
  self.cache_position
}

///|
pub fn LunaPagedProfileRow::page_table_start(self : LunaPagedProfileRow) -> Int {
  self.page_table_start
}

///|
pub fn LunaPagedProfileRow::block_count(self : LunaPagedProfileRow) -> Int {
  self.block_count
}

///|
/// Caller-asserted digest of the raw page-table bytes observed by the offline
/// profiler. It is trace identity only: it authenticates neither device-page
/// ownership nor execution operands.
pub struct LunaPagedPageTableTraceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaPagedPageTableTraceDigest::from_sha256(
  value : String,
) -> LunaPagedPageTableTraceDigest raise LunaKernelProfileError {
  if !luna_kernel_profile_is_sha256(value) {
    raise LunaKernelProfileInvalid(Digest)
  }
  LunaPagedPageTableTraceDigest(value)
}

///|
pub fn LunaPagedPageTableTraceDigest::as_hex(
  self : LunaPagedPageTableTraceDigest,
) -> String {
  self.0
}

///|
/// Exact live step shape. The page-table trace claim binds the raw profiler
/// bytes while the rows bind their non-overlapping slices and geometry. The
/// claim never proves physical page identity, ownership, or execution.
pub struct LunaPagedProfileShape {
  priv tokens_per_page : Int
  priv prefill_row_count : Int
  priv decode_row_count : Int
  priv query_token_count : Int
  priv page_table_block_count : Int
  priv page_table_trace_digest : LunaPagedPageTableTraceDigest
  priv rows : ReadOnlyArray[LunaPagedProfileRow]
  priv digest : LunaKernelProfileDigest
} derive(Debug, Eq)

///|
pub fn LunaPagedProfileShape::new(
  tokens_per_page~ : Int,
  prefill_row_count~ : Int,
  decode_row_count~ : Int,
  query_token_count~ : Int,
  page_table_block_count~ : Int,
  page_table_trace_digest~ : LunaPagedPageTableTraceDigest,
  rows : ArrayView[LunaPagedProfileRow],
) -> LunaPagedProfileShape raise LunaKernelProfileError {
  if tokens_per_page <= 0 ||
    tokens_per_page > 1_048_576 ||
    prefill_row_count < 0 ||
    prefill_row_count > 8192 ||
    decode_row_count < 0 ||
    decode_row_count > 8192 ||
    query_token_count <= 0 ||
    query_token_count > 2_147_483_647 ||
    page_table_block_count <= 0 {
    raise LunaKernelProfileInvalid(Shape)
  }
  let row_count = prefill_row_count + decode_row_count
  if row_count <= 0 || row_count != rows.length() || row_count > 8192 {
    raise LunaKernelProfileInvalid(Shape)
  }
  let mut observed_prefill = 0
  let mut observed_decode = 0
  let mut observed_query = 0
  let mut observed_blocks = 0
  let mut saw_decode = false
  for row in rows {
    match row.phase() {
      PagedPrefill => {
        if saw_decode {
          raise LunaKernelProfileInvalid(RowOrder)
        }
        observed_prefill += 1
      }
      PagedDecode => {
        saw_decode = true
        observed_decode += 1
      }
    }
    if observed_query > 2_147_483_647 - row.query_length() ||
      observed_blocks > 2_147_483_647 - row.block_count() ||
      row.page_table_start() != observed_blocks {
      raise LunaKernelProfileInvalid(Shape)
    }
    let base_blocks = row.context_length() / tokens_per_page
    let expected_blocks = if row.context_length() % tokens_per_page == 0 {
      base_blocks
    } else {
      base_blocks + 1
    }
    if row.block_count() != expected_blocks {
      raise LunaKernelProfileMismatch(PageTable)
    }
    observed_query += row.query_length()
    observed_blocks += row.block_count()
  }
  if observed_prefill != prefill_row_count ||
    observed_decode != decode_row_count ||
    observed_query != query_token_count ||
    observed_blocks != page_table_block_count {
    raise LunaKernelProfileMismatch(Shape)
  }
  let owned_rows = ReadOnlyArray::from_array(rows.to_owned())
  let digest = digest_luna_paged_profile_shape(
    tokens_per_page, prefill_row_count, decode_row_count, query_token_count, page_table_block_count,
    page_table_trace_digest, owned_rows,
  )
  {
    tokens_per_page,
    prefill_row_count,
    decode_row_count,
    query_token_count,
    page_table_block_count,
    page_table_trace_digest,
    rows: owned_rows,
    digest,
  }
}

///|
pub fn LunaPagedProfileShape::tokens_per_page(
  self : LunaPagedProfileShape,
) -> Int {
  self.tokens_per_page
}

///|
pub fn LunaPagedProfileShape::prefill_row_count(
  self : LunaPagedProfileShape,
) -> Int {
  self.prefill_row_count
}

///|
pub fn LunaPagedProfileShape::decode_row_count(
  self : LunaPagedProfileShape,
) -> Int {
  self.decode_row_count
}

///|
pub fn LunaPagedProfileShape::query_token_count(
  self : LunaPagedProfileShape,
) -> Int {
  self.query_token_count
}

///|
pub fn LunaPagedProfileShape::page_table_block_count(
  self : LunaPagedProfileShape,
) -> Int {
  self.page_table_block_count
}

///|
pub fn LunaPagedProfileShape::page_table_trace_digest(
  self : LunaPagedProfileShape,
) -> LunaPagedPageTableTraceDigest {
  self.page_table_trace_digest
}

///|
pub fn LunaPagedProfileShape::rows(
  self : LunaPagedProfileShape,
) -> ReadOnlyArray[LunaPagedProfileRow] {
  self.rows
}

///|
pub fn LunaPagedProfileShape::digest(
  self : LunaPagedProfileShape,
) -> LunaKernelProfileDigest {
  self.digest
}

///|
/// Derived view of one profile in an admitted paged launch-contract set.
///
/// There is deliberately no public constructor. Admission derives every field
/// from the opaque launch authority; profiler input cannot assemble target,
/// scope, capacity, or KV-layout claims.
pub struct LunaPagedProfileContract {
  priv scope : @launch_contract.PagedContractScope
  priv catalog_version : @catalog.CatalogVersion
  priv kernel_profile : @launch_contract.PagedKernelProfile
  priv device_kv_layout : @device_layout.DeviceKvLayout
  priv launches : ReadOnlyArray[@launch_contract.PagedKvAotLaunchContract]
  priv launch_identity : LunaKernelProfileDigest
} derive(Debug, Eq)

///|
pub fn LunaPagedProfileContract::scope(
  self : LunaPagedProfileContract,
) -> @launch_contract.PagedContractScope {
  self.scope
}

///|
pub fn LunaPagedProfileContract::catalog_version(
  self : LunaPagedProfileContract,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn LunaPagedProfileContract::kernel_profile(
  self : LunaPagedProfileContract,
) -> @launch_contract.PagedKernelProfile {
  self.kernel_profile
}

///|
pub fn LunaPagedProfileContract::device_kv_layout(
  self : LunaPagedProfileContract,
) -> @device_layout.DeviceKvLayout {
  self.device_kv_layout
}

///|
/// Digest of the complete selected launch records, including content-addressed
/// entry points, dimensions, ordered operands and workspace claims.
pub fn LunaPagedProfileContract::launch_identity(
  self : LunaPagedProfileContract,
) -> LunaKernelProfileDigest {
  self.launch_identity
}

///|
pub fn LunaPagedProfileContract::launches(
  self : LunaPagedProfileContract,
) -> ReadOnlyArray[@launch_contract.PagedKvAotLaunchContract] {
  self.launches
}

///|
pub fn LunaPagedProfileContract::kv_head_count(
  self : LunaPagedProfileContract,
) -> Int {
  self.device_kv_layout.geometry().kv_head_count()
}

///|
pub fn LunaPagedProfileContract::head_dimension(
  self : LunaPagedProfileContract,
) -> Int {
  self.device_kv_layout.geometry().head_dimension()
}

///|
pub fn LunaPagedProfileContract::kv_dtype(
  self : LunaPagedProfileContract,
) -> @spec.ModelDType {
  self.device_kv_layout.geometry().dtype()
}

///|
pub fn LunaPagedProfileContract::tokens_per_page(
  self : LunaPagedProfileContract,
) -> Int {
  self.device_kv_layout.tokens_per_page()
}

///|
pub fn LunaPagedProfileContract::total_page_count(
  self : LunaPagedProfileContract,
) -> Int {
  self.device_kv_layout.total_page_count()
}

///|
pub fn TensorParallelKvPlan::identity(
  self : TensorParallelKvPlan,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelKvPlan::generation_value(
  self : TensorParallelKvPlan,
) -> UInt64 {
  self.generation_value
}

///|
pub fn TensorParallelKvPlan::world_size(self : TensorParallelKvPlan) -> Int {
  self.world_size
}

///|
pub fn TensorParallelKvPlan::global_kv_head_count(
  self : TensorParallelKvPlan,
) -> Int {
  self.global_kv_head_count
}

///|
pub fn TensorParallelKvPlan::local_kv_head_count(
  self : TensorParallelKvPlan,
) -> Int {
  self.local_kv_head_count
}

///|
pub fn TensorParallelKvPlan::layer_count(self : TensorParallelKvPlan) -> Int {
  self.local_layout.geometry().layer_count()
}

///|
pub fn TensorParallelKvPlan::head_dimension(self : TensorParallelKvPlan) -> Int {
  self.local_layout.geometry().head_dimension()
}

///|
pub fn TensorParallelKvPlan::layout_version(self : TensorParallelKvPlan) -> Int {
  self.local_layout.stable_layout_version()
}

///|
pub fn TensorParallelKvPlan::tokens_per_page(
  self : TensorParallelKvPlan,
) -> Int {
  self.local_layout.tokens_per_page()
}

///|
pub fn TensorParallelKvPlan::total_page_count(
  self : TensorParallelKvPlan,
) -> Int {
  self.local_layout.total_page_count()
}

///|
pub fn TensorParallelKvPlan::page_logical_bytes(
  self : TensorParallelKvPlan,
) -> Int64 {
  self.local_layout.page_logical_bytes()
}

///|
pub fn TensorParallelKvPlan::page_stride_bytes(
  self : TensorParallelKvPlan,
) -> Int64 {
  self.local_layout.page_stride_bytes()
}

///|
pub fn TensorParallelKvPlan::rank_arena_bytes(
  self : TensorParallelKvPlan,
) -> Int64 {
  self.local_layout.arena_bytes()
}

///|
/// Returns the exact canonical immutable layout shared by one admitted rank.
/// The value contains byte geometry only and confers no allocation authority.
pub fn TensorParallelKvPlan::rank_layout(
  self : TensorParallelKvPlan,
  rank : Int,
) -> @device_layout.DeviceKvLayout raise TensorParallelKvPlanError {
  if rank < 0 || rank >= self.world_size {
    raise rejected(Rank, rank)
  }
  self.local_layout
}

///|
pub fn TensorParallelKvPlan::full_arena_bytes(
  self : TensorParallelKvPlan,
) -> Int64 {
  self.full_arena_bytes
}

///|
pub fn TensorParallelKvPlan::rank_kv_reservation_bytes(
  self : TensorParallelKvPlan,
  rank : Int,
) -> Int64 raise TensorParallelKvPlanError {
  if rank < 0 || rank >= self.world_size {
    raise rejected(Rank, rank)
  }
  self.rank_kv_reservations[rank]
}

///|
pub fn TensorParallelKvPlan::rank_binding(
  self : TensorParallelKvPlan,
  rank : Int,
) -> TensorParallelKvRankBinding raise TensorParallelKvPlanError {
  if rank < 0 || rank >= self.world_size {
    raise rejected(Rank, rank)
  }
  self.rank_bindings[rank]
}

///|
/// Rejects a stale or substituted worker model generation without mutation.
pub fn TensorParallelKvPlan::require_current_generation(
  self : TensorParallelKvPlan,
  generation : @worker_protocol.ModelPlanGeneration,
) -> Unit raise TensorParallelKvPlanError {
  if generation.value() != self.generation_value {
    raise rejected(ModelGeneration, -1)
  }
}

///|
/// Re-authenticates the complete model/generation/layout discriminator before
/// a later startup owner publishes rank bindings.
pub fn TensorParallelKvPlan::require_current_contract(
  self : TensorParallelKvPlan,
  identity : @spec.ModelIdentity,
  generation : @worker_protocol.ModelPlanGeneration,
  layout_version : Int,
) -> Unit raise TensorParallelKvPlanError {
  if identity != self.identity {
    raise rejected(ModelIdentity, -1)
  }
  self.require_current_generation(generation)
  if layout_version != self.layout_version() {
    raise rejected(LayoutVersion, -1)
  }
}

///|
fn TensorParallelKvPlan::validate_rank_and_page(
  self : TensorParallelKvPlan,
  rank : Int,
  page : @page_allocator.PageId,
) -> Unit raise TensorParallelKvPlanError {
  if rank < 0 || rank >= self.world_size {
    raise rejected(Rank, rank)
  }
  if page.index() < 0 || page.index() >= self.total_page_count() {
    raise rejected(PageIndex, rank)
  }
}

///|
/// Projects the exact allocator-issued logical page into one rank's arena.
/// The returned scalar retains the page index and generation unchanged.
pub fn TensorParallelKvPlan::page_region(
  self : TensorParallelKvPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  rank : Int,
  page : @page_allocator.PageId,
  layer_index : Int,
  component : @device_layout.DeviceKvComponent,
) -> TensorParallelKvPageRegion raise TensorParallelKvPlanError {
  self.require_current_generation(generation)
  self.validate_rank_and_page(rank, page)
  let region = self.local_layout.region(layer_index, component, page.index()) catch {
    _ => raise rejected(LayerIndex, rank)
  }
  {
    rank,
    page_index: page.index(),
    page_generation: page.generation(),
    offset_bytes: region.offset_bytes(),
    logical_bytes: region.logical_bytes(),
    reserved_bytes: region.reserved_bytes(),
  }
}

///|
/// Resolves one rank-local BF16 element without allocating or mutating page
/// metadata. All ranks use the same logical page index and generation.
pub fn TensorParallelKvPlan::element_offset_bytes(
  self : TensorParallelKvPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  rank : Int,
  page : @page_allocator.PageId,
  layer_index : Int,
  component : @device_layout.DeviceKvComponent,
  token_index : Int,
  local_kv_head_index : Int,
  head_dimension_index : Int,
) -> Int64 raise TensorParallelKvPlanError {
  self.require_current_generation(generation)
  self.validate_rank_and_page(rank, page)
  if layer_index < 0 || layer_index >= self.layer_count() {
    raise rejected(LayerIndex, rank)
  }
  if token_index < 0 || token_index >= self.tokens_per_page() {
    raise rejected(TokenIndex, rank)
  }
  if local_kv_head_index < 0 || local_kv_head_index >= self.local_kv_head_count {
    raise rejected(LocalKvHeadIndex, rank)
  }
  if head_dimension_index < 0 || head_dimension_index >= self.head_dimension() {
    raise rejected(HeadDimensionIndex, rank)
  }
  self.local_layout.element_offset_bytes(
    layer_index,
    component,
    page.index(),
    token_index,
    local_kv_head_index,
    head_dimension_index,
  ) catch {
    _ => raise rejected(SizeOverflow, rank)
  }
}

///|
pub fn TensorParallelKvPageRegion::rank(
  self : TensorParallelKvPageRegion,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelKvPageRegion::page_index(
  self : TensorParallelKvPageRegion,
) -> Int {
  self.page_index
}

///|
pub fn TensorParallelKvPageRegion::page_generation(
  self : TensorParallelKvPageRegion,
) -> Int {
  self.page_generation
}

///|
pub fn TensorParallelKvPageRegion::offset_bytes(
  self : TensorParallelKvPageRegion,
) -> Int64 {
  self.offset_bytes
}

///|
pub fn TensorParallelKvPageRegion::logical_bytes(
  self : TensorParallelKvPageRegion,
) -> Int64 {
  self.logical_bytes
}

///|
pub fn TensorParallelKvPageRegion::reserved_bytes(
  self : TensorParallelKvPageRegion,
) -> Int64 {
  self.reserved_bytes
}

///|
pub fn TensorParallelKvContractDigest::as_sha256(
  self : TensorParallelKvContractDigest,
) -> String {
  self.sha256
}

///|
pub fn TensorParallelKvRankBinding::identity(
  self : TensorParallelKvRankBinding,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelKvRankBinding::generation_value(
  self : TensorParallelKvRankBinding,
) -> UInt64 {
  self.generation_value
}

///|
pub fn TensorParallelKvRankBinding::rank(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelKvRankBinding::world_size(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelKvRankBinding::process_ordinal(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.process_ordinal
}

///|
pub fn TensorParallelKvRankBinding::target(
  self : TensorParallelKvRankBinding,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn TensorParallelKvRankBinding::global_kv_head_count(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.global_kv_head_count
}

///|
pub fn TensorParallelKvRankBinding::local_kv_head_count(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.local_kv_head_count
}

///|
pub fn TensorParallelKvRankBinding::layer_count(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.layer_count
}

///|
pub fn TensorParallelKvRankBinding::head_dimension(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.head_dimension
}

///|
pub fn TensorParallelKvRankBinding::layout_version(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.layout_version
}

///|
pub fn TensorParallelKvRankBinding::tokens_per_page(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.tokens_per_page
}

///|
pub fn TensorParallelKvRankBinding::total_page_count(
  self : TensorParallelKvRankBinding,
) -> Int {
  self.total_page_count
}

///|
pub fn TensorParallelKvRankBinding::page_logical_bytes(
  self : TensorParallelKvRankBinding,
) -> Int64 {
  self.page_logical_bytes
}

///|
pub fn TensorParallelKvRankBinding::page_stride_bytes(
  self : TensorParallelKvRankBinding,
) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn TensorParallelKvRankBinding::component_stride_bytes(
  self : TensorParallelKvRankBinding,
) -> Int64 {
  self.component_stride_bytes
}

///|
pub fn TensorParallelKvRankBinding::layer_stride_bytes(
  self : TensorParallelKvRankBinding,
) -> Int64 {
  self.layer_stride_bytes
}

///|
pub fn TensorParallelKvRankBinding::arena_bytes(
  self : TensorParallelKvRankBinding,
) -> Int64 {
  self.arena_bytes
}

///|
pub fn TensorParallelKvRankBinding::kv_reservation_bytes(
  self : TensorParallelKvRankBinding,
) -> Int64 {
  self.kv_reservation_bytes
}

///|
pub fn TensorParallelKvRankBinding::digest(
  self : TensorParallelKvRankBinding,
) -> TensorParallelKvContractDigest {
  self.digest
}

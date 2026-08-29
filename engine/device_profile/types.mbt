///|
/// Semantic request phase, separate from the computational kernel shape.
pub(all) enum ExecutionPhase {
  FullPrefill
  FullRecompute
} derive(Debug, Eq)

///|
/// Exact request dimensions consumed by one prepared dispatch profile.
///
/// Both supported phases use all `query_tokens_per_row` tokens as both query
/// and context. `FullRecompute` makes the current phase-1 generation contract
/// explicit: every step recomputes the complete sequence because this graph
/// has no KV-cache input or paged-attention state.
pub struct ExactExecutionShape {
  priv phase : ExecutionPhase
  priv batch_rows : Int
  priv query_tokens_per_row : Int
  priv context_tokens : Int
  priv token_rows : Int64
} derive(Debug, Eq)

///|
/// Exact computational dimensions that an admitted kernel profile must match.
///
/// Full prefill and full recompute share this key when their dimensions are
/// identical because both execute the same stateless full-sequence graph.
pub struct ExactKernelShape {
  priv batch_rows : Int
  priv sequence_tokens : Int
  priv token_rows : Int64
} derive(Debug, Eq)

///|
/// Stable physical allocation class for an exact region view.
pub(all) enum ExecutionBufferSpace {
  TokenInput
  WeightArena
  ActivationWorkspaceArena
} derive(Debug, Eq)

///|
/// Stable integer region identity within one `DeviceExecutionProfile`.
pub struct ProfileRegionId {
  priv value : Int
} derive(Debug, Eq)

///|
/// One immutable logical slice of a reusable device allocation.
///
/// `logical_bytes` is the exact request requirement. `reserved_bytes` is the
/// capacity already planned for its buffer space and is never expanded during
/// dispatch.
pub struct ExecutionRegionView {
  priv id : ProfileRegionId
  priv space : ExecutionBufferSpace
  priv offset_bytes : Int64
  priv logical_bytes : Int64
  priv reserved_bytes : Int64
  priv alignment_bytes : Int64
} derive(Debug, Eq)

///|
/// Exact activation value metadata used to validate operation dataflow.
pub struct ActivationValueView {
  priv reference : @plan.ActivationRef
  priv width : Int64
  priv token_rows : Int64
  priv region : ExecutionRegionView
} derive(Debug, Eq)

///|
/// One exact operation input, preserving the semantic input order.
pub(all) enum OperationInputView {
  TokenIds(ExecutionRegionView)
  Weight(@plan.TensorRef, ExecutionRegionView)
  Activation(ActivationValueView)
} derive(Debug, Eq)

///|
/// Immutable, allocation-free dispatch metadata for one operation.
///
/// `kernel_shape` must be matched exactly by the selected AOT/vendor execution
/// plan. The maximum memory-envelope shape is not a compatible substitute.
pub struct OperationExecutionProfile {
  priv operation_id : @plan.OperationId
  priv kernel_shape : ExactKernelShape
  priv inputs : ReadOnlyArray[OperationInputView]
  priv outputs : ReadOnlyArray[ActivationValueView]
  priv workspace : ExecutionRegionView
} derive(Debug, Eq)

///|
/// Semantic interpretation of the graph's terminal activation.
pub(all) enum FinalOutputSemantics {
  LanguageModelLogits
  ActivationOutput
} derive(Debug, Eq)

///|
/// One bounded row subview into the terminal activation region.
///
/// This is intentionally not an `ExecutionRegionView`: its backing region ID
/// retains one exact physical-region meaning while the row carries a narrower
/// offset and length.
pub struct FinalRowView {
  priv batch_index : Int
  priv backing_region : ProfileRegionId
  priv offset_bytes : Int64
  priv byte_length : Int64
  priv alignment_bytes : Int64
} derive(Debug, Eq)

///|
/// Exact terminal output plus one precomputed terminal row per batch item.
///
/// For language-model logits, `selected_rows` are the last query-token logits
/// used by sampling. For other graph outputs they are the last query-token
/// activation rows. Multi-batch prefill rows may be non-contiguous and are
/// therefore represented individually.
pub struct FinalOutputView {
  priv semantics : FinalOutputSemantics
  priv value : ActivationValueView
  priv row_bytes : Int64
  priv selected_rows : ReadOnlyArray[FinalRowView]
} derive(Debug, Eq)

///|
/// Immutable exact-shape bridge from startup plans to prepared dispatch.
///
/// Building this value may allocate its bounded metadata arrays. Reading its
/// operation, input, output, workspace, and terminal views does not allocate.
pub struct DeviceExecutionProfile {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv shape : ExactExecutionShape
  priv token_staging : ExecutionRegionView
  priv token_staging_capacity_bytes : Int64
  priv regions : ReadOnlyArray[ExecutionRegionView]
  priv operations : ReadOnlyArray[OperationExecutionProfile]
  priv final_output : FinalOutputView
} derive(Debug, Eq)

///|
pub fn ExactExecutionShape::full_prefill(
  batch_rows~ : Int,
  sequence_tokens~ : Int,
) -> ExactExecutionShape raise DeviceProfileError {
  exact_shape(FullPrefill, batch_rows, sequence_tokens, sequence_tokens)
}

///|
pub fn ExactExecutionShape::full_recompute(
  batch_rows~ : Int,
  sequence_tokens~ : Int,
) -> ExactExecutionShape raise DeviceProfileError {
  exact_shape(FullRecompute, batch_rows, sequence_tokens, sequence_tokens)
}

///|
pub fn ExactExecutionShape::phase(self : ExactExecutionShape) -> ExecutionPhase {
  self.phase
}

///|
pub fn ExactExecutionShape::batch_rows(self : ExactExecutionShape) -> Int {
  self.batch_rows
}

///|
pub fn ExactExecutionShape::query_tokens_per_row(
  self : ExactExecutionShape,
) -> Int {
  self.query_tokens_per_row
}

///|
pub fn ExactExecutionShape::context_tokens(self : ExactExecutionShape) -> Int {
  self.context_tokens
}

///|
pub fn ExactExecutionShape::token_rows(self : ExactExecutionShape) -> Int64 {
  self.token_rows
}

///|
pub fn ExactExecutionShape::kernel_shape(
  self : ExactExecutionShape,
) -> ExactKernelShape {
  {
    batch_rows: self.batch_rows,
    sequence_tokens: self.query_tokens_per_row,
    token_rows: self.token_rows,
  }
}

///|
pub fn ExactKernelShape::batch_rows(self : ExactKernelShape) -> Int {
  self.batch_rows
}

///|
pub fn ExactKernelShape::sequence_tokens(self : ExactKernelShape) -> Int {
  self.sequence_tokens
}

///|
pub fn ExactKernelShape::token_rows(self : ExactKernelShape) -> Int64 {
  self.token_rows
}

///|
pub fn ExecutionPhase::as_stable_int(self : ExecutionPhase) -> Int {
  match self {
    FullPrefill => 0
    FullRecompute => 1
  }
}

///|
pub fn ProfileRegionId::as_int(self : ProfileRegionId) -> Int {
  self.value
}

///|
pub fn ProfileRegionId::from_stable_int(
  value : Int,
) -> ProfileRegionId raise DeviceProfileError {
  checked_region_id(Int64::from_int(value))
}

///|
pub fn ExecutionRegionView::id(self : ExecutionRegionView) -> ProfileRegionId {
  self.id
}

///|
pub fn ExecutionRegionView::space(
  self : ExecutionRegionView,
) -> ExecutionBufferSpace {
  self.space
}

///|
pub fn ExecutionRegionView::offset_bytes(self : ExecutionRegionView) -> Int64 {
  self.offset_bytes
}

///|
pub fn ExecutionRegionView::logical_bytes(self : ExecutionRegionView) -> Int64 {
  self.logical_bytes
}

///|
pub fn ExecutionRegionView::reserved_bytes(self : ExecutionRegionView) -> Int64 {
  self.reserved_bytes
}

///|
pub fn ExecutionRegionView::alignment_bytes(
  self : ExecutionRegionView,
) -> Int64 {
  self.alignment_bytes
}

///|
pub fn ExecutionRegionView::end_offset_bytes(
  self : ExecutionRegionView,
) -> Int64 {
  self.offset_bytes + self.logical_bytes
}

///|
pub fn ActivationValueView::reference(
  self : ActivationValueView,
) -> @plan.ActivationRef {
  self.reference
}

///|
pub fn ActivationValueView::width(self : ActivationValueView) -> Int64 {
  self.width
}

///|
pub fn ActivationValueView::token_rows(self : ActivationValueView) -> Int64 {
  self.token_rows
}

///|
pub fn ActivationValueView::region(
  self : ActivationValueView,
) -> ExecutionRegionView {
  self.region
}

///|
pub fn OperationExecutionProfile::operation_id(
  self : OperationExecutionProfile,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn OperationExecutionProfile::kernel_shape(
  self : OperationExecutionProfile,
) -> ExactKernelShape {
  self.kernel_shape
}

///|
pub fn OperationExecutionProfile::inputs(
  self : OperationExecutionProfile,
) -> ReadOnlyArray[OperationInputView] {
  self.inputs
}

///|
pub fn OperationExecutionProfile::outputs(
  self : OperationExecutionProfile,
) -> ReadOnlyArray[ActivationValueView] {
  self.outputs
}

///|
pub fn OperationExecutionProfile::workspace(
  self : OperationExecutionProfile,
) -> ExecutionRegionView {
  self.workspace
}

///|
pub fn FinalOutputView::semantics(
  self : FinalOutputView,
) -> FinalOutputSemantics {
  self.semantics
}

///|
pub fn FinalOutputView::value(self : FinalOutputView) -> ActivationValueView {
  self.value
}

///|
pub fn FinalOutputView::row_bytes(self : FinalOutputView) -> Int64 {
  self.row_bytes
}

///|
pub fn FinalOutputView::selected_rows(
  self : FinalOutputView,
) -> ReadOnlyArray[FinalRowView] {
  self.selected_rows
}

///|
pub fn FinalRowView::batch_index(self : FinalRowView) -> Int {
  self.batch_index
}

///|
pub fn FinalRowView::backing_region(self : FinalRowView) -> ProfileRegionId {
  self.backing_region
}

///|
pub fn FinalRowView::offset_bytes(self : FinalRowView) -> Int64 {
  self.offset_bytes
}

///|
pub fn FinalRowView::byte_length(self : FinalRowView) -> Int64 {
  self.byte_length
}

///|
pub fn FinalRowView::alignment_bytes(self : FinalRowView) -> Int64 {
  self.alignment_bytes
}

///|
pub fn FinalRowView::end_offset_bytes(self : FinalRowView) -> Int64 {
  self.offset_bytes + self.byte_length
}

///|
pub fn DeviceExecutionProfile::identity(
  self : DeviceExecutionProfile,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeviceExecutionProfile::target(
  self : DeviceExecutionProfile,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeviceExecutionProfile::shape(
  self : DeviceExecutionProfile,
) -> ExactExecutionShape {
  self.shape
}

///|
pub fn DeviceExecutionProfile::token_staging(
  self : DeviceExecutionProfile,
) -> ExecutionRegionView {
  self.token_staging
}

///|
pub fn DeviceExecutionProfile::token_staging_capacity_bytes(
  self : DeviceExecutionProfile,
) -> Int64 {
  self.token_staging_capacity_bytes
}

///|
pub fn DeviceExecutionProfile::region_count(
  self : DeviceExecutionProfile,
) -> Int {
  self.regions.length()
}

///|
pub fn DeviceExecutionProfile::region(
  self : DeviceExecutionProfile,
  id : ProfileRegionId,
) -> ExecutionRegionView? {
  for region in self.regions {
    if region.id() == id {
      return Some(region)
    }
  }
  None
}

///|
pub fn DeviceExecutionProfile::operation_count(
  self : DeviceExecutionProfile,
) -> Int {
  self.operations.length()
}

///|
pub fn DeviceExecutionProfile::operation(
  self : DeviceExecutionProfile,
  id : @plan.OperationId,
) -> OperationExecutionProfile? {
  self.operations.get(id.as_int())
}

///|
pub fn DeviceExecutionProfile::final_output(
  self : DeviceExecutionProfile,
) -> FinalOutputView {
  self.final_output
}

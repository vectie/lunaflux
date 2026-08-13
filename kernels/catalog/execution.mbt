///|
/// Complete semantic ABI attached to one catalog-v2 operation entry.
///
/// The decoder-layer identity is semantic. Persistent KV is passed as exact
/// per-layer Key and Value component bases; ordered page metadata indexes the
/// bound layout without specializing on total physical page count.
pub struct KernelExecutionSemantics {
  kv_layer : @plan.KvLayerId?
  runtime_inputs : ReadOnlyArray[@plan.OperationRuntimeInput]
  persistent_state : @plan.PersistentStateAccess
  kv_layout : PagedKvLayoutContract?
} derive(Debug, Eq)

///|
/// Capacity-independent physical KV indexing contract for catalog v2.
///
/// Total physical page count is deliberately absent: it sizes admitted
/// component spans but must not specialize kernel indexing.
pub struct PagedKvLayoutContract {
  layout : @plan.KvCacheLayout
  stable_version : Int
  tokens_per_page : Int
} derive(Debug, Eq)

///|
pub fn PagedKvLayoutContract::new(
  layout~ : @plan.KvCacheLayout,
  stable_version~ : Int,
  tokens_per_page~ : Int,
) -> PagedKvLayoutContract raise CatalogError {
  if stable_version <= 0 ||
    stable_version != layout.stable_version() ||
    tokens_per_page <= 0 {
    raise InvalidCatalog(InvalidExecutionSemantics)
  }
  { layout, stable_version, tokens_per_page }
}

///|
pub fn PagedKvLayoutContract::layout(
  self : PagedKvLayoutContract,
) -> @plan.KvCacheLayout {
  self.layout
}

///|
pub fn PagedKvLayoutContract::stable_version(
  self : PagedKvLayoutContract,
) -> Int {
  self.stable_version
}

///|
pub fn PagedKvLayoutContract::tokens_per_page(
  self : PagedKvLayoutContract,
) -> Int {
  self.tokens_per_page
}

///|
/// Defensively owns one catalog-v2 live-input contract.
///
/// Operation-specific legality is checked by `CatalogEntry::new_paged_v2`.
pub fn KernelExecutionSemantics::paged_v2(
  kv_layer~ : @plan.KvLayerId?,
  runtime_inputs~ : ArrayView[@plan.OperationRuntimeInput],
  persistent_state~ : @plan.PersistentStateAccess,
  kv_layout~ : PagedKvLayoutContract?,
) -> KernelExecutionSemantics raise CatalogError {
  match persistent_state {
    NoPersistentState =>
      if kv_layout is Some(_) {
        raise InvalidCatalog(InvalidExecutionSemantics)
      }
    KvCacheReadWrite(state_layer) => {
      guard kv_layer is Some(operation_layer) && operation_layer == state_layer else {
        raise InvalidCatalog(InvalidExecutionSemantics)
      }
      if kv_layout is None {
        raise InvalidCatalog(InvalidExecutionSemantics)
      }
    }
  }
  {
    kv_layer,
    runtime_inputs: ReadOnlyArray::from_array(runtime_inputs),
    persistent_state,
    kv_layout,
  }
}

///|
pub fn KernelExecutionSemantics::kv_layer(
  self : KernelExecutionSemantics,
) -> @plan.KvLayerId? {
  self.kv_layer
}

///|
pub fn KernelExecutionSemantics::runtime_inputs(
  self : KernelExecutionSemantics,
) -> ReadOnlyArray[@plan.OperationRuntimeInput] {
  self.runtime_inputs
}

///|
pub fn KernelExecutionSemantics::persistent_state(
  self : KernelExecutionSemantics,
) -> @plan.PersistentStateAccess {
  self.persistent_state
}

///|
pub fn KernelExecutionSemantics::kv_layout(
  self : KernelExecutionSemantics,
) -> PagedKvLayoutContract? {
  self.kv_layout
}

///|
fn runtime_inputs_match(
  actual : ReadOnlyArray[@plan.OperationRuntimeInput],
  expected : ArrayView[@plan.OperationRuntimeInput],
) -> Bool {
  if actual.length() != expected.length() {
    return false
  }
  for index, value in actual {
    if value != expected[index] {
      return false
    }
  }
  true
}

///|
fn paged_semantics_are_legal(
  kind : @plan.OperationKind,
  semantics : KernelExecutionSemantics,
) -> Bool {
  match kind {
    RotaryEmbedding =>
      semantics.kv_layer is Some(_) &&
      runtime_inputs_match(semantics.runtime_inputs, [
        StepCounts,
        QueryPositions,
      ]) &&
      semantics.persistent_state == NoPersistentState &&
      semantics.kv_layout is None
    CausalAttention =>
      match semantics.kv_layer {
        Some(layer) =>
          runtime_inputs_match(semantics.runtime_inputs, [
            StepCounts,
            QueryPositions,
            QueryRowOffsets,
            SequenceLengths,
            PageTableOffsets,
            PageTablePageIndices,
          ]) &&
          semantics.persistent_state == KvCacheReadWrite(layer) &&
          (match semantics.kv_layout {
            Some(contract) =>
              contract.layout == LayerMajorSplitKeyValueV1 &&
              contract.stable_version == 1
            None => false
          })
        None => false
      }
    _ =>
      semantics.kv_layer is None &&
      runtime_inputs_match(semantics.runtime_inputs, [StepCounts]) &&
      semantics.persistent_state == NoPersistentState &&
      semantics.kv_layout is None
  }
}

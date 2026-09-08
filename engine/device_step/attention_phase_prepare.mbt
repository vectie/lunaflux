///|
/// Immutable startup substitution. Partial kernels narrow their read domain;
/// only the final attention writer adds the disjoint decode companion.
priv struct PagedAttentionPhaseReplacement {
  operation_id : @plan.OperationId
  function_index : Int
  prefill_arguments : @device.KernelArguments
  decode_companion : PreparedPagedGraphStep?
}

///|
fn paged_attention_phase_replacement(
  replacements : ArrayView[PagedAttentionPhaseReplacement],
  step : PreparedPagedGraphStep,
) -> PagedAttentionPhaseReplacement? {
  for replacement in replacements {
    if replacement.operation_id == step.operation_id &&
      replacement.function_index == step.function_index {
      return Some(replacement)
    }
  }
  None
}

///|
fn paged_attention_decode_companion(
  resources : PagedGraphResources,
  operation : @plan.OperationId,
) -> PreparedPagedGraphStep? {
  for replacement in resources.attention_phase_replacements {
    if replacement.operation_id == operation &&
      replacement.decode_companion is Some(companion) {
      return Some(companion)
    }
  }
  None
}

///|
fn register_attention_phase_replacement(
  resources : PagedGraphResources,
  prefill : PreparedPagedGraphStep,
  decode : PreparedPagedGraphStep?,
) -> Unit raise DeviceStepError {
  if paged_attention_phase_replacement(
      resources.attention_phase_replacements[:],
      prefill,
    )
    is Some(_) {
    raise InvalidExecutor(GraphBucket)
  }
  resources.attention_phase_replacements.push({
    operation_id: prefill.operation_id,
    function_index: prefill.function_index,
    prefill_arguments: prefill.arguments,
    decode_companion: decode,
  })
}

///|
/// A pure list rewrite: all effects before the replaced attention stay in
/// place; prefill and decode write disjoint original output rows, then the
/// original downstream consumer executes once. This runs only at startup.
fn[Step] specialize_attention_phase_list(
  steps : ReadOnlyArray[Step],
  replacement : (Step) -> (Step, Step?)?,
) -> ReadOnlyArray[Step]? {
  let output : Array[Step] = []
  let mut changed = false
  for step in steps {
    match replacement(step) {
      Some((prefill, decode)) => {
        changed = true
        output.push(prefill)
        if decode is Some(companion) {
          output.push(companion)
        }
      }
      None => output.push(step)
    }
  }
  if changed {
    Some(ReadOnlyArray::from_array(output))
  } else {
    None
  }
}

///|
fn mixed_attention_steps(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
) -> ReadOnlyArray[PreparedPagedGraphStep]? {
  specialize_attention_phase_list(steps, step => {
    match
      paged_attention_phase_replacement(
        resources.attention_phase_replacements[:],
        step,
      ) {
      Some(replacement) =>
        Some(
          (
            { ..step, arguments: replacement.prefill_arguments },
            replacement.decode_companion,
          ),
        )
      None => None
    }
  })
}

///|
fn select_paged_attention_phase_owner(
  resources : PagedGraphResources,
  owner_index : Int,
  phase : @luna_execution_graph_strategy.ExecutionGraphPhase,
) -> Int {
  let mixed = resources.ordered_executors[owner_index].mixed_owner_index
  if phase == MixedGraph && mixed >= 0 {
    mixed
  } else {
    owner_index
  }
}

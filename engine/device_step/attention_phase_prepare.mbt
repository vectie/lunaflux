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
fn[T] find_mixed_attention_item(
  items : ArrayView[T],
  matches : (T) -> Bool,
) -> T? {
  for item in items {
    if matches(item) {
      return Some(item)
    }
  }
  None
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
  replacement : (Step) -> (Step, ReadOnlyArray[Step])?,
) -> ReadOnlyArray[Step]? {
  let output : Array[Step] = []
  let mut changed = false
  for step in steps {
    match replacement(step) {
      Some((prefill, decode)) => {
        changed = true
        output.push(prefill)
        for companion in decode {
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
  partitioned_decode? : Bool = false,
) -> ReadOnlyArray[PreparedPagedGraphStep]? {
  // Reject an incomplete alternative before materializing any graph. Never
  // silently mix measured split choices with ordinary decode companions.
  if partitioned_decode {
    let mut writers = 0
    for step in steps {
      if paged_attention_phase_replacement(
          resources.attention_phase_replacements[:],
          step,
        )
        is Some(replacement) &&
        replacement.decode_companion is Some(_) {
        writers += 1
        if !resources.attention_partitioned_decode_companions.any(item => {
            item.operation_id == step.operation_id
          }) {
          return None
        }
      }
    }
    if writers == 0 {
      return None
    }
  }
  specialize_attention_phase_list(steps, step => {
    match
      paged_attention_phase_replacement(
        resources.attention_phase_replacements[:],
        step,
      ) {
      Some(replacement) => {
        let companions : ReadOnlyArray[PreparedPagedGraphStep] = match
          replacement.decode_companion {
          None => []
          Some(decode) =>
            if partitioned_decode {
              let split = find_mixed_attention_item(
                resources.attention_partitioned_decode_companions[:],
                item => item.operation_id == step.operation_id,
              ).unwrap()
              [split.partial, split.merge]
            } else {
              [decode]
            }
        }
        Some(
          ({ ..step, arguments: replacement.prefill_arguments, }, companions),
        )
      }
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

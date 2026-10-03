///|
/// Materialize a bounded mixed-only executor while retaining the exact
/// baseline fallback. Arguments, launches, and ownership are fixed at startup.
fn prepare_mixed_attention_executor(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  launches : ArrayView[@device.OrderedKernelLaunch],
  resources : PagedGraphResources,
  owner_index : Int,
) -> Unit raise DeviceStepError {
  if prepare_mixed_attention_variant(
      context, steps, launches, resources, owner_index,
    )
    is Some(mixed) {
    resources.ordered_executors[owner_index].mixed_owner_index = mixed
  }
}

///|
/// Both alternatives retain the same bucket-specialized upstream launches.
/// Only the disjoint decode writer changes; no additional ingress, KV write,
/// projection, or sampling is replayed.
fn prepare_mixed_attention_variant(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  launches : ArrayView[@device.OrderedKernelLaunch],
  resources : PagedGraphResources,
  owner_index : Int,
  partitioned_decode? : Bool = false,
) -> Int? raise DeviceStepError {
  if steps.length() != launches.length() {
    raise InvalidExecutor(GraphBucket)
  }
  guard mixed_attention_steps(steps, resources, partitioned_decode~)
    is Some(mixed) else {
    return None
  }
  let owner = resources.ordered_executors[owner_index]
  let captured = owner.resource.execution_mode() == OrderedKernelCaptured
  let reservation = if captured {
    resources.effect_capture_budget.reserve(mixed.length())
  } else {
    None
  }
  if captured && reservation is None {
    return None
  }
  guard resources.stream is Some(stream) && stream.open else {
    raise InvalidExecutor(ResourceState)
  }
  let selected : Array[@device.OrderedKernelLaunch] = []
  for index, step in steps {
    match
      paged_attention_phase_replacement(
        resources.attention_phase_replacements[:],
        step,
      ) {
      Some(replacement) => {
        selected.push(
          launches[index].with_arguments(replacement.prefill_arguments),
        )
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
        for decode in companions {
          guard resources.functions.get(decode.function_index) is Some(function) &&
            function.open else {
            raise InvalidExecutor(ResourceState)
          }
          selected.push(
            @device.OrderedKernelLaunch::new(
              function.resource,
              decode.dimensions,
              decode.arguments,
            ),
          )
        }
      }
      None => selected.push(launches[index])
    }
  }
  let operation_ids = ReadOnlyArray::from_array(
    mixed.map(step => step.operation_id),
  )
  let executor = context.create_ordered_kernel_executor(
    stream.resource,
    selected,
    policy=if captured {
      OrderedKernelCaptureWithEagerFallback
    } else {
      OrderedKernelEagerOnly
    },
  ) catch {
    error => {
      if captured {
        return None
      }
      raise ExecutorDevice(OrderedExecutor, error)
    }
  }
  let mixed_index = resources.ordered_executors.length()
  resources.ordered_executors.push({
    resource: executor,
    operation_ids,
    effect_only_owner_index: -1,
    mixed_owner_index: -1,
    open: true,
  })
  if captured && executor.execution_mode() != OrderedKernelCaptured {
    executor.close() catch {
      error => raise ExecutorDevice(OrderedExecutor, error)
    }
    resources.ordered_executors[mixed_index].open = false
    return None
  }
  if reservation is Some(remaining) {
    resources.effect_capture_budget = remaining
  }
  // A MixedGraph necessarily contains a producing decode row, so an
  // output-free suffix variant can never be selected for this owner.
  Some(mixed_index)
}

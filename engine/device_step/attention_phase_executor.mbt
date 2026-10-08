///|
/// Materialize a bounded mixed-only executor while retaining the exact
/// baseline fallback. Arguments, launches, and ownership are fixed at startup.
fn prepare_mixed_attention_executor(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  launches : ArrayView[@device.OrderedKernelLaunch],
  resources : PagedGraphResources,
  owner_index : Int,
  projection_query_bound? : Int = 0,
) -> Unit raise DeviceStepError {
  if prepare_mixed_attention_variant(
      context,
      steps,
      launches,
      resources,
      owner_index,
      projection_query_bound~,
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
  capture_budget_reserved? : Bool = false,
  decode_row_bound? : Int? = None,
  projection_query_bound? : Int = 0,
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
  let reservation = if captured && !capture_budget_reserved {
    paged_variant_capture_budget(resources, projection_query_bound).reserve(
      mixed.length(),
    )
  } else {
    None
  }
  if captured && !capture_budget_reserved && reservation is None {
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
          let dimensions = if partitioned_decode &&
            decode_row_bound is Some(bound) {
            let split = find_mixed_attention_item(
              resources.attention_partitioned_decode_companions[:],
              item => item.operation_id == step.operation_id,
            ).unwrap()
            let admitted = if decode.function_index ==
              split.partial.function_index {
              split.partial_dimensions
            } else {
              split.merge_dimensions
            }
            paged_device_launch_dimensions(
              paged_capped_query_token_launch(admitted, bound),
            )
          } else {
            decode.dimensions
          }
          selected.push(
            @device.OrderedKernelLaunch::new(
              function.resource,
              dimensions,
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
    projection_owner_index: -1,
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
    update_paged_variant_capture_budget(
      resources, projection_query_bound, remaining,
    )
  }
  // A MixedGraph necessarily contains a producing decode row. Its projection
  // companion retains that output; an output-free suffix is never selected.
  prepare_paged_vendor_projection_executor(
    context,
    ReadOnlyArray::from_array(mixed),
    selected[:],
    resources,
    mixed_index,
    projection_query_bound,
  )
  Some(mixed_index)
}

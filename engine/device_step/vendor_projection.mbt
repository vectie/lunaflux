///|
/// A pure operand/shape plan, independent of CUDA launch geometry and symbols.
/// Only the terminal device lowering chooses the vendor implementation.
priv struct PagedVendorProjectionRule {
  operation : @plan.OperationId
  function_index : Int
  columns : Int
  inner : Int
  input : Int
  weight : Int
  output : Int
}

///|
fn paged_vendor_projection_rules(
  blueprint : PagedKvExecutionBlueprint,
  functions : FixedArray[Int],
  companions : FixedArray[Int],
  rows : Int,
) -> ReadOnlyArray[PagedVendorProjectionRule] raise DeviceStepError {
  if rows == 0 {
    return []
  }
  if rows < 2 ||
    rows > blueprint.profile().max_query_tokens() ||
    blueprint.graph_memory_evidence().availability() != GraphMemoryAbsent {
    raise InvalidExecutor(GraphBucket)
  }
  let rules : Array[PagedVendorProjectionRule] = []
  for ordinal, step in blueprint.steps() {
    let down = step.operation_kind() == GatedMlp && companions[ordinal] >= 0
    if step.operation_kind() != OutputProjection && !down {
      continue
    }
    let mut input = -1
    let mut output = -1
    let weights : Array[Int] = []
    for index, operand in step.operands() {
      if (down && operand.role() == Workspace) ||
        (!down && operand.role() is ActivationInput(_)) {
        if input >= 0 {
          raise InvalidExecutor(GraphBucket)
        }
        input = index
      }
      if operand.role() is ActivationOutput(_) {
        if output >= 0 {
          raise InvalidExecutor(GraphBucket)
        }
        output = index
      }
      if operand.role() is WeightInput(_) {
        weights.push(index)
      }
    }
    // The admitted dense projection ABI has one weight; the split gated MLP
    // ABI orders gate, up, down. Never replace the gate/up operation itself.
    if input < 0 || output < 0 || weights.length() != (if down { 3 } else { 1 }) {
      raise InvalidExecutor(GraphBucket)
    }
    let row_bytes = blueprint.profile().max_query_tokens().to_int64() * 2L
    let x_bytes = step.operands()[input].byte_count()
    let y_bytes = step.operands()[output].byte_count()
    if x_bytes % row_bytes != 0L ||
      y_bytes % row_bytes != 0L ||
      x_bytes / row_bytes <= 0L ||
      y_bytes / row_bytes <= 0L ||
      x_bytes / row_bytes > 2147483647L ||
      y_bytes / row_bytes > 2147483647L {
      raise InvalidExecutor(GraphBucket)
    }
    let inner = (x_bytes / row_bytes).to_int()
    let columns = (y_bytes / row_bytes).to_int()
    let weight = weights[weights.length() - 1]
    if step.operands()[weight].byte_count() !=
      inner.to_int64() * columns.to_int64() * 2L {
      raise InvalidExecutor(GraphBucket)
    }
    rules.push({
      operation: step.operation_id(),
      function_index: if down {
        companions[ordinal]
      } else {
        functions[ordinal]
      },
      columns,
      inner,
      input,
      weight,
      output,
    })
  }
  if rules.is_empty() {
    raise InvalidExecutor(GraphBucket)
  }
  ReadOnlyArray::from_array(rules)
}

///|
fn prepare_paged_vendor_projection_executor(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  launches : ArrayView[@device.OrderedKernelLaunch],
  resources : PagedGraphResources,
  original : Int,
  query_bound : Int,
) -> Unit raise DeviceStepError {
  let rows = resources.vendor_projection_exact_rows
  if rows == 0 || query_bound != rows {
    return
  }
  if steps.length() != launches.length() {
    raise InvalidExecutor(GraphBucket)
  }
  let owner = resources.ordered_executors[original]
  if owner.resource.execution_mode() != OrderedKernelCaptured {
    raise InvalidExecutor(GraphBucket)
  }
  let records : Array[@device.OrderedBf16Projection] = []
  for index, step in steps {
    for rule in resources.vendor_projection_rules {
      if rule.operation == step.operation_id &&
        rule.function_index == step.function_index {
        records.push(
          @device.OrderedBf16Projection::new(
            launch_index=index,
            rows~,
            columns=rule.columns,
            inner=rule.inner,
            input_argument=rule.input,
            weight_argument=rule.weight,
            output_argument=rule.output,
          ) catch {
            error => raise ExecutorDevice(OrderedExecutor, error)
          },
        )
      }
    }
  }
  if records.is_empty() {
    return
  }
  guard resources.stream is Some(stream) &&
    stream.open &&
    resources.vendor_capture_budget.reserve(launches.length())
    is Some(remaining) else {
    raise InvalidExecutor(GraphBucket)
  }
  let executor = context.create_ordered_kernel_executor(
    stream.resource,
    launches,
    policy=OrderedKernelCaptureRequired,
    projections=records[:],
  ) catch {
    error => raise ExecutorDevice(OrderedExecutor, error)
  }
  let index = resources.ordered_executors.length()
  resources.ordered_executors.push({
    resource: executor,
    operation_ids: owner.operation_ids,
    effect_only_owner_index: -1,
    mixed_owner_index: -1,
    projection_owner_index: -1,
    open: true,
  })
  resources.vendor_capture_budget = remaining
  owner.projection_owner_index = index
}

///|
fn vendor_projection_matches(rows : Int, summary : DeviceStepSummary) -> Bool {
  rows > 1 && summary.prefill_count > 0 && summary.token_count == rows
}

///|
/// Allocation-free exact-domain dispatch after attention and output-demand
/// selection. Decode and partially filled buckets retain their AOT executor.
fn select_paged_vendor_projection(
  resources : PagedGraphResources,
  original : Int,
  summary : DeviceStepSummary,
) -> Int {
  let candidate = resources.ordered_executors[original].projection_owner_index
  if candidate >= 0 &&
    vendor_projection_matches(resources.vendor_projection_exact_rows, summary) {
    candidate
  } else {
    original
  }
}

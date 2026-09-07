///|
/// Resolve all variant symbols at startup. Full envelopes without an explicit
/// row bound keep the original entry; smaller captured envelopes select the
/// smallest covering variant. No module lookup or selection occurs per token.
fn prepare_projection_variants(
  blueprint : PagedKvExecutionBlueprint,
  functions : ReadOnlyArray[PagedFunctionBlueprint],
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  for step in blueprint.steps() {
    if step.row_variants.is_empty() {
      continue
    }
    if step.operation_kind() != QkvProjection &&
      step.operation_kind() != OutputProjection {
      raise executor_operation_error(step.operation_id(), GraphBucket)
    }
    let find = fn(symbol : String) -> Int raise DeviceStepError {
      for index, function in functions {
        if function.module_index == step.artifact_module_index() &&
          function.symbol == symbol {
          return index
        }
      }
      raise InvalidExecutor(ArtifactModule)
    }
    let base = find(step.function_symbol())
    if resources.projection_variants.any(v => v.base_function_index == base) {
      continue
    }
    for variant in step.row_variants {
      if variant.maximum_token_rows() > blueprint.profile().max_query_tokens() {
        raise executor_operation_error(step.operation_id(), GraphBucket)
      }
      let source = variant.dimensions()
      let width = paged_projection_output_width(blueprint, step)
      let launch = if source.block_x() >= 32 &&
        source.block_x() <= 512 &&
        (source.block_x() & (source.block_x() - 1)) == 0 &&
        source.shared_memory_bytes() == source.block_x() * 32 {
        ProjectionTensorCoreGridX(source, width)
      } else {
        ProjectionScalarGridX(source, width)
      }
      resources.projection_variants.push({
        base_function_index: base,
        maximum_token_rows: variant.maximum_token_rows(),
        function_index: find(variant.function_symbol()),
        launch,
      })
    }
  }
}

///|
fn select_projection_variant(
  variants : ArrayView[PagedProjectionVariant],
  base_function_index : Int,
  query_token_bound : Int,
) -> PagedProjectionVariant? {
  if query_token_bound <= 0 {
    return None
  }
  let mut selected : PagedProjectionVariant? = None
  for variant in variants {
    if variant.base_function_index == base_function_index &&
      variant.maximum_token_rows >= query_token_bound {
      match selected {
        Some(previous) if previous.maximum_token_rows <=
          variant.maximum_token_rows => ()
        _ => selected = Some(variant)
      }
    }
  }
  selected
}

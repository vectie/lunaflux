///|
fn paged_function_index(
  functions : Array[PagedFunctionBlueprint],
  identity : @catalog.AotKernelEntryPoint,
) -> Int32? {
  for index, function in functions {
    if function.identity == identity {
      return Some(index)
    }
  }
  None
}

///|
fn paged_function_symbol_index(
  functions : Array[PagedFunctionBlueprint],
  module_index : Int32,
  symbol : StringView,
) -> Int32? {
  for index, function in functions {
    if function.module_index == module_index &&
      function.symbol == symbol.to_owned() {
      return Some(index)
    }
  }
  None
}

///|
fn paged_step_has_split_decode_mlp(step : PagedKvExecutionStep) -> Bool {
  step.operation_kind() == GatedMlp &&
  step.operands().any(operand => operand.role() == Workspace)
}

///|
fn collect_paged_functions(
  blueprint : PagedKvExecutionBlueprint,
) -> (
  ReadOnlyArray[PagedFunctionBlueprint],
  FixedArray[Int32],
  FixedArray[Int32],
) {
  let functions : Array[PagedFunctionBlueprint] = []
  let indices = FixedArray::make(blueprint.steps().length(), 0)
  let split_decode_indices = FixedArray::make(blueprint.steps().length(), -1)
  for step_index, step in blueprint.steps() {
    match paged_function_index(functions, step.entry_point()) {
      Some(index) => indices[step_index] = index
      None => {
        functions.push({
          identity: step.entry_point(),
          module_index: step.artifact_module_index(),
          symbol: step.function_symbol(),
        })
        indices[step_index] = functions.length() - 1
      }
    }
  }
  for step_index, step in blueprint.steps() {
    if paged_step_has_split_decode_mlp(step) {
      let symbol = step.function_symbol() + "_down"
      match
        paged_function_symbol_index(
          functions,
          step.artifact_module_index(),
          symbol,
        ) {
        Some(index) => split_decode_indices[step_index] = index
        None => {
          functions.push({
            identity: step.entry_point(),
            module_index: step.artifact_module_index(),
            symbol,
          })
          split_decode_indices[step_index] = functions.length() - 1
        }
      }
    }
  }
  for step in blueprint.steps() {
    for variant in step.row_variants {
      if paged_step_has_split_decode_mlp(step) {
        let symbol = variant.function_symbol() + "_down"
        if paged_function_symbol_index(
            functions,
            step.artifact_module_index(),
            symbol,
          )
          is None {
          functions.push({
            identity: step.entry_point(),
            module_index: step.artifact_module_index(),
            symbol,
          })
        }
      }
      if paged_function_symbol_index(
          functions,
          step.artifact_module_index(),
          variant.function_symbol(),
        )
        is None {
        functions.push({
          identity: step.entry_point(),
          module_index: step.artifact_module_index(),
          symbol: variant.function_symbol(),
        })
      }
    }
  }
  (ReadOnlyArray::from_array(functions), indices, split_decode_indices)
}

///|
fn load_paged_modules(
  context : @device.Context,
  artifacts : @artifact.KernelArtifactBundle,
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  for image in artifacts.modules() {
    let resource = context.load_module(image.module_bytes()) catch {
      error => raise ExecutorDevice(Module, error)
    }
    resources.modules.push({ resource, open: true, })
  }
}

///|
fn load_paged_functions(
  functions : ReadOnlyArray[PagedFunctionBlueprint],
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  for function in functions {
    guard resources.modules.get(function.module_index) is Some(owner) else {
      raise InvalidExecutor(ArtifactModule)
    }
    let resource = owner.resource.load_function(@utf8.encode(function.symbol)) catch {
      error => raise ExecutorDevice(Function, error)
    }
    resources.functions.push({
      resource,
      module_index: function.module_index,
      open: true,
    })
  }
}

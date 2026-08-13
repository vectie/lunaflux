///|
const RECIPE_MAX_CONFIG_BYTES : Int64 = 16777216L

///|
const RECIPE_MAX_WEIGHT_HEADER_BYTES : Int = 16777216

///|
const RECIPE_JSON_DEPTH : Int = 16

///|
const RECIPE_MAX_OPERATIONS : Int = 4096

///|
const RECIPE_MAX_SYMBOL_BYTES : Int = 1024

///|
const RECIPE_MAX_RANK : Int = 2

///|
const RECIPE_MAX_TENSOR_NAME_BYTES : Int = 256

///|
const RECIPE_TRANSFER_CHUNK_BYTES : Int = 4194304

///|
const RECIPE_ALIGNMENT_BYTES : Int64 = 256L

///|
const RECIPE_MAX_OPERANDS : Int = 32

///|
const RECIPE_MAX_CANONICAL_BYTES : Int = 16777216

///|
fn exact_graph_count(layer_count : Int) -> Int raise DeviceWorkerBootstrapError {
  if layer_count <= 0 || layer_count > (RECIPE_MAX_OPERATIONS - 3) / 9 {
    raise Invalid(OperationCount)
  }
  layer_count * 9 + 3
}

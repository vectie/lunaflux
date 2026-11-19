///|
const RECIPE_MAX_CONFIG_BYTES : Int64 = 16777216L

///|
const RECIPE_MAX_WEIGHT_HEADER_BYTES : Int32 = 16777216

///|
const RECIPE_JSON_DEPTH : Int32 = 16

///|
const RECIPE_MAX_OPERATIONS : Int32 = 4096

///|
const RECIPE_MAX_SYMBOL_BYTES : Int32 = 1024

///|
const RECIPE_MAX_RANK : Int32 = 2

///|
const RECIPE_MAX_TENSOR_NAME_BYTES : Int32 = 256

///|
const RECIPE_TRANSFER_CHUNK_BYTES : Int32 = 4194304

///|
const RECIPE_ALIGNMENT_BYTES : Int64 = 256L

///|
const RECIPE_MAX_OPERANDS : Int32 = 32

///|
const RECIPE_MAX_CANONICAL_BYTES : Int32 = 16777216

///|
fn exact_graph_count(
  layer_count : Int32,
) -> Int32 raise DeviceWorkerBootstrapError {
  if layer_count <= 0 || layer_count > (RECIPE_MAX_OPERATIONS - 3) / 9 {
    raise Invalid(OperationCount)
  }
  layer_count * 9 + 3
}

///|
/// One bounded alternative for the same numeric/operand ABI. Selection is
/// materialized into execution buckets at startup, never interpreted per token.
pub struct ReusableFusedRuntimeRowVariant {
  priv maximum_token_rows : Int
  priv query_tile_rows : Int
  priv symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv module_bytes : Bytes
} derive(Eq)

///|
pub fn ReusableFusedRuntimeRowVariant::new(
  maximum_token_rows~ : Int,
  query_tile_rows~ : Int,
  function_symbol~ : String,
  launch~ : @launch_contract.AotLaunchDimensions,
  module_bytes~ : Bytes,
) -> ReusableFusedRuntimeRowVariant raise FusedArtifactAdmissionError {
  if maximum_token_rows < 2 ||
    maximum_token_rows > 1048576 ||
    query_tile_rows <= 0 ||
    query_tile_rows > maximum_token_rows ||
    launch.grid_x() != (maximum_token_rows - 1) / query_tile_rows + 1 {
    raise InvalidFusedArtifact(Launch)
  }
  reusable_runtime_validate_module(
    AttentionIngressPreparedRotaryProductionV11,
    0,
    4,
    function_symbol,
    launch,
    module_bytes,
  )
  {
    maximum_token_rows,
    query_tile_rows,
    symbol: function_symbol,
    launch,
    module_bytes,
  }
}

///|
pub fn ReusableFusedRuntimeRowVariant::maximum_token_rows(
  self : ReusableFusedRuntimeRowVariant,
) -> Int {
  self.maximum_token_rows
}

///|
pub fn ReusableFusedRuntimeRowVariant::query_tile_rows(
  self : ReusableFusedRuntimeRowVariant,
) -> Int {
  self.query_tile_rows
}

///|
pub fn ReusableFusedRuntimeRowVariant::function_symbol(
  self : ReusableFusedRuntimeRowVariant,
) -> String {
  self.symbol
}

///|
pub fn ReusableFusedRuntimeRowVariant::launch_dimensions(
  self : ReusableFusedRuntimeRowVariant,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn ReusableFusedRuntimeRowVariant::module_bytes(
  self : ReusableFusedRuntimeRowVariant,
) -> Bytes {
  self.module_bytes
}

///|
pub fn ApprovedReusableFusedRuntimeModule::row_variant(
  self : ApprovedReusableFusedRuntimeModule,
) -> ReusableFusedRuntimeRowVariant? {
  self.row_variant
}

///|
fn reusable_runtime_validate_row_variant(
  abi : ReusableFusedRuntimeKernelAbi,
  symbol : String,
  launch : @launch_contract.AotLaunchDimensions,
  heads_per_cta : Int,
  variant : ReusableFusedRuntimeRowVariant?,
) -> Unit raise FusedArtifactAdmissionError {
  guard variant is Some(v) else { return }
  // Explicitly closed to the cached-rotary ingress ABI. Other families must
  // supply their own numeric/operand equivalence contract before extension.
  if abi != AttentionIngressPreparedRotaryProductionV11 ||
    v.symbol != symbol ||
    v.launch.grid_y() != launch.grid_y() ||
    v.launch.block_x() != launch.block_x() ||
    heads_per_cta != 1 ||
    v.launch.shared_memory_bytes() > 98304 {
    raise InvalidFusedArtifact(KernelAbi)
  }
}

///|
fn reusable_runtime_append_row_variant(
  out : StringBuilder,
  prefix : String,
  variant : ReusableFusedRuntimeRowVariant?,
) -> Unit raise FusedArtifactAdmissionError {
  out.write_string(
    prefix +
    "row_variant_count=" +
    (if variant is Some(_) { "1" } else { "0" }) +
    "\n",
  )
  guard variant is Some(v) else { return }
  let prefix = prefix + "row_variant_"
  for
    (key, value) in [
      ("maximum_token_rows", v.maximum_token_rows),
      ("query_tile_rows", v.query_tile_rows),
      ("grid_x", v.launch.grid_x()),
      ("grid_y", v.launch.grid_y()),
      ("grid_z", v.launch.grid_z()),
      ("block_x", v.launch.block_x()),
      ("block_y", v.launch.block_y()),
      ("block_z", v.launch.block_z()),
      ("shared_bytes", v.launch.shared_memory_bytes()),
    ] {
    out.write_string(prefix + key + "=" + value.to_string() + "\n")
  }
  out.write_string(prefix + "function_symbol=" + v.symbol + "\n")
  out.write_string(
    prefix +
    "module_sha256=" +
    fused_artifact_digest(v.module_bytes).as_hex() +
    "\n",
  )
  out.write_string(
    prefix + "module_hex=" + @crypto.bytes_to_hex_string(v.module_bytes) + "\n",
  )
}

///|
fn reusable_runtime_read_row_variant(
  reader : FusedDeploymentReader,
  prefix : String,
) -> ReusableFusedRuntimeRowVariant? raise FusedArtifactAdmissionError {
  match reader.int(prefix + "row_variant_count") {
    0 => return None
    1 => ()
    _ => raise InvalidFusedArtifact(Binding)
  }
  let prefix = prefix + "row_variant_"
  let maximum = reader.int(prefix + "maximum_token_rows")
  let rows = reader.int(prefix + "query_tile_rows")
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=reader.int(prefix + "grid_x"),
    grid_y=reader.int(prefix + "grid_y"),
    grid_z=reader.int(prefix + "grid_z"),
    block_x=reader.int(prefix + "block_x"),
    block_y=reader.int(prefix + "block_y"),
    block_z=reader.int(prefix + "block_z"),
    shared_memory_bytes=reader.int(prefix + "shared_bytes"),
  ) catch {
    _ => raise InvalidFusedArtifact(Launch)
  }
  let symbol = reader.value(prefix + "function_symbol")
  ignore(fused_deployment_digest(reader.value(prefix + "module_sha256")))
  let bytes = fused_deployment_decode_hex(reader.value(prefix + "module_hex"))
  if bytes.is_empty() {
    raise MismatchedFusedArtifact(Module)
  }
  Some(
    ReusableFusedRuntimeRowVariant::new(
      maximum_token_rows=maximum,
      query_tile_rows=rows,
      function_symbol=symbol,
      launch~,
      module_bytes=bytes,
    ),
  )
}

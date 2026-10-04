///|
/// Compact weights enter the existing scheduled/physical projection compiler,
/// rather than an independent low-bit GEMM implementation. This ABI is not
/// interchangeable with a BF16 weight pointer or a native quantized MMA ABI.
pub struct PrecisionProjectionSource {
  priv source : Bytes
  priv precision : @precision.LinearPrecision
  priv physical : @luna_projection_tile_compiler.PhysicalProjectionTileCompilation
  priv grid : Int
  priv threads : Int
}

///|
pub fn precision_projection_source(
  physical : @luna_projection_tile_compiler.PhysicalProjectionTileCompilation,
  precision : @precision.LinearPrecision,
  symbol~ : String,
) -> PrecisionProjectionSource raise ProjectionAotError {
  if symbol.length() == 0 || symbol.length() > 80 {
    raise Invalid(Source)
  }
  for i, c in symbol {
    if !(c is ('a'..='z' | 'A'..='Z' | '_')) && !(i > 0 && c is ('0'..='9')) {
      raise Invalid(Source)
    }
  }
  if precision.route() != DecodeToBf16 || precision.accumulator() != F32 {
    raise Unsupported(CompilerPolicy)
  }
  if physical.program().epilogue() != ProjectionStoreRows ||
    !(physical.program().shape().family() is (Dense | LanguageModelHead)) {
    raise Unsupported(Operation)
  }
  let (activation, weight, _) = precision.operands()
  let (maximum_tokens, input_width) = activation.representation().shape()
  let (output_width, weight_width) = weight.representation().shape()
  if weight_width != input_width ||
    physical.program().shape().dimensions() != (input_width, output_width, 0) {
    raise Mismatch(Shape)
  }
  guard physical.matrix is Some(matrix) else { raise Unsupported(Strategy) }
  if matrix.schedule.minimum_rows() > 2 || matrix.owners_per_group != 32 {
    raise Unsupported(Strategy)
  }
  let region = matrix.operands.region(1).unwrap()
  let (_, weight_rows, tile_columns) = region.geometry()
  let (_, scratch) = weight.memory()
  if scratch <
    weight_rows.to_int64() *
    tile_columns.to_int64() *
    matrix.schedule.stage_count().to_int64() *
    2L {
    raise LimitExceeded(OperandSize)
  }
  let refined = weight.refine_region(region) catch {
    _ => raise Invalid(OperandSize)
  }
  let selected = physical.schedule().demand() == QueryRowEnds
  let parameters : ProjectionParameters = {
    family: if selected {
      LanguageModelHead
    } else {
      DenseProjection
    },
    input_width,
    output_width,
    query_width: 0,
    intermediate_width: 0,
  }
  let groups = physical.schedule().output_tiles_per_workgroup()
  let matrix_columns = matrix.schedule.output_tiles() * 16
  let rows = matrix.schedule.row_width()
  let matrix_grid = (1L + (maximum_tokens.to_int64() - 1L) / rows.to_int64()) *
    (1L + (output_width.to_int64() - 1L) / matrix_columns.to_int64())
  let grid = if selected {
    1L + (output_width.to_int64() - 1L) / (groups.to_int64() * 16L)
  } else {
    matrix_grid.max(1L + (output_width.to_int64() - 1L) / groups.to_int64())
  }
  let threads = region.ownership().geometry().1
  if grid > 2147483647L || threads > 1024 {
    raise LimitExceeded(LaunchDimensions)
  }
  {
    source: projection_tensor_core_direct_dense_source(
      symbol,
      parameters,
      maximum_tokens,
      maximum_tokens,
      physical,
      precision=refined,
    ),
    precision,
    physical,
    grid: grid.to_int(),
    threads,
  }
}

///|
pub fn PrecisionProjectionSource::source_bytes(
  self : PrecisionProjectionSource,
) -> Bytes {
  self.source
}

///|
pub fn PrecisionProjectionSource::launch(
  self : PrecisionProjectionSource,
) -> (Int, Int) {
  (self.grid, self.threads)
}

///|
pub fn PrecisionProjectionSource::precision(
  self : PrecisionProjectionSource,
) -> @precision.LinearPrecision {
  self.precision
}

///|
pub fn PrecisionProjectionSource::physical(
  self : PrecisionProjectionSource,
) -> @luna_projection_tile_compiler.PhysicalProjectionTileCompilation {
  self.physical
}

///|
/// One AOT specialization sharing the parent entry's operands and numerical
/// contract. The bound describes token rows, not context or output rows.
pub struct KernelRowVariant {
  priv maximum_token_rows : Int
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv companion_dimensions : @launch_contract.AotLaunchDimensions?
  priv bounded_work_grid_x : Int?
} derive(Debug, Eq)

///|
pub fn KernelRowVariant::new(
  maximum_token_rows~ : Int,
  function_symbol~ : String,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  companion_dimensions? : @launch_contract.AotLaunchDimensions? = None,
  bounded_work_grid_x? : Int? = None,
) -> KernelRowVariant {
  {
    maximum_token_rows,
    function_symbol,
    dimensions,
    companion_dimensions,
    bounded_work_grid_x,
  }
}

///|
/// Optional compiler-derived work cardinality for this bounded executable.
/// It preserves the published launch's block/storage contract. Absent values
/// retain legacy launch planning; no execution layer reconstructs a schedule.
pub fn KernelRowVariant::bounded_work_grid_x(self : KernelRowVariant) -> Int? {
  self.bounded_work_grid_x
}

///|
pub fn KernelRowVariant::maximum_token_rows(self : KernelRowVariant) -> Int {
  self.maximum_token_rows
}

///|
pub fn KernelRowVariant::function_symbol(self : KernelRowVariant) -> String {
  self.function_symbol
}

///|
pub fn KernelRowVariant::dimensions(
  self : KernelRowVariant,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
fn validate_row_variants(
  input : KernelEntryPointInput,
  maximum_symbol_bytes : Int,
) -> Unit raise KernelArtifactError {
  if input.row_variants.length() > 19 {
    raise InvalidRowVariants(input.entry_point)
  }
  let mut previous = 0
  for index, variant in input.row_variants {
    if variant.bounded_work_grid_x is Some(grid) &&
      (grid <= 0 || grid > variant.dimensions.grid_x()) {
      raise InvalidRowVariants(input.entry_point)
    }
    let bound = variant.maximum_token_rows
    if bound <= previous ||
      bound > 1048576 ||
      (bound != 1 && (bound < 8 || (bound & (bound - 1)) != 0)) {
      raise InvalidRowVariants(input.entry_point)
    }
    validate_symbol(
      input.entry_point,
      variant.function_symbol,
      maximum_symbol_bytes,
    )
    if variant.function_symbol == input.function_symbol ||
      input.row_variants[:index].any(other => {
        other.function_symbol == variant.function_symbol
      }) {
      raise InvalidRowVariants(input.entry_point)
    }
    previous = bound
  }
}

///|
// Both directions and compact cache rotation use the same numerical lowering.
// The caller declares left/right/cosine/sine, output and index.
fn rotary_product_source(product : @precision.RotaryProductLaw) -> String {
  let bindings = ["left", "right", "cosine", "sine"]
  let real = @arithmetic.scalar_expression(
    product.expression(imaginary=false),
    bindings,
  ) catch {
    _ => abort("invalid rotary program")
  }
  let imaginary = @arithmetic.scalar_expression(
    product.expression(imaginary=true),
    bindings,
  ) catch {
    _ => abort("invalid rotary program")
  }
  (
    $|    output[index] = \{real};
    $|    output[index + 1ULL] = \{imaginary};
  )
}

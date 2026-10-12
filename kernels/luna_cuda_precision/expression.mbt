///|
/// Render a validated pure expression. Binding strings are private lowering
/// operands, not user source. No algebraic simplification or fast-math choices
/// occur here. Primitive names do not leak back into the numeric program.
pub fn scalar_expression(
  program : @precision.NumericExpression,
  bindings : ArrayView[String],
) -> String raise @precision.PrecisionPlanError {
  guard bindings.length() == program.inputs().length() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  fn render(expr : @precision.NumericExpr) -> String {
    match expr {
      Input(index) => bindings[index]
      Float32Bits(0U) => "0.0f"
      Float32Bits(0x3f800000U) => "1.0f"
      Float32Bits(0xff800000U) => "-INFINITY"
      Float32Bits(bits) => "__uint_as_float(\{bits}U)"
      Binary(op, a, b) => {
        let name = match op {
          AddRN => "__fadd_rn"
          SubtractRN => "__fsub_rn"
          MultiplyRN => "__fmul_rn"
          DivideRN => "__fdiv_rn"
          MaximumNumber => "fmaxf"
        }
        "\{name}(\{render(a)}, \{render(b)})"
      }
      Unary(Negate, value) => "-(\{render(value)})"
      Unary(op, value) => {
        let name = match op {
          NaturalExp => "expf"
          ReciprocalSqrt => "rsqrtf"
          SquareRoot => "sqrtf"
          Negate => abort("handled above")
        }
        "\{name}(\{render(value)})"
      }
      FusedMultiplyAdd(a, b, c) =>
        "__fmaf_rn(\{render(a)}, \{render(b)}, \{render(c)})"
      Convert(kind, value) => {
        // Same-type conversions are exact identity, including NaN payloads;
        // avoid a spurious roundtrip through a different storage type.
        fn scalar(expr : @precision.NumericExpr) -> @precision.NumericScalar {
          match expr {
            Input(i) => program.inputs()[i].0
            Convert(k, _) => k
            _ => Float32
          }
        }
        if scalar(value) == kind {
          render(value)
        } else {
          let name = match kind {
            Float32 => "__bfloat162float"
            BFloat16 => "__float2bfloat16_rn"
          }
          "\{name}(\{render(value)})"
        }
      }
    }
  }
  render(program.simplify_exact().root())
}

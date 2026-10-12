///|
/// acc[0] = initial F32 bits; acc[i+1] = update(acc[i], operands[i]).
/// Iteration is ascending with exactly extent updates, no reassociation,
/// implicit reduction tree or hidden cast. Input zero is the accumulator.
/// This small compositional form coexists with the existing explicit tree/dot
/// laws; it does not reclassify those alternative laws as equivalent folds.
pub struct OrderedNumericFold {
  extent : Int
  initial_bits : UInt
  update : NumericExpression
}

///|
pub fn OrderedNumericFold::new(
  extent~ : Int,
  initial_bits~ : UInt,
  update~ : NumericExpression,
) -> Self raise PrecisionPlanError {
  guard extent > 0 &&
    update.output == Float32 &&
    update.inputs.length() > 0 &&
    update.inputs[0] == (Float32, AnyFloat) else {
    raise IllegalPlacement
  }
  { extent, initial_bits, update, }
}

///|
pub fn LearnedGatedPoolPrecision::fold(
  self : Self,
  operation : GatedPoolExpression,
) -> OrderedNumericFold raise PrecisionPlanError {
  OrderedNumericFold::new(
    extent=self.reduction_extent(operation),
    initial_bits=match operation {
      Maximum => 0xff800000U
      _ => 0U
    },
    update=self.expression(operation),
  )
}

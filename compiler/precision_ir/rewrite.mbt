///|
/// A closed, deliberately small rule set. No measurement, implementation name,
/// claimed proof flag, or caller-supplied error bound can enter this analysis.
pub(all) enum NumericRelation {
  IdenticalProgram
  ExactFiniteBf16Roundtrip
  BoundedBf16Rounding(absolute_error~ : Double)
  DifferentProgram
} derive(Eq)

///|
pub extend NumericRelation with Eq::{equal, not_equal}

///|
fn NumericExpr::exact_normal_form(
  self : Self,
  inputs : ReadOnlyArray[(NumericScalar, NumericDomain)],
) -> Self {
  match self {
    Convert(kind, value) => {
      let normalized = value.exact_normal_form(inputs)
      match (kind, normalized) {
        (BFloat16, Convert(Float32, Input(index))) if inputs[index].0 ==
          BFloat16 &&
          inputs[index].1 != AnyFloat => Input(index)
        _ => Convert(kind, normalized)
      }
    }
    Unary(op, value) => Unary(op, value.exact_normal_form(inputs))
    Binary(op, a, b) =>
      Binary(op, a.exact_normal_form(inputs), b.exact_normal_form(inputs))
    FusedMultiplyAdd(a, b, c) =>
      FusedMultiplyAdd(
        a.exact_normal_form(inputs),
        b.exact_normal_form(inputs),
        c.exact_normal_form(inputs),
      )
    Input(_) | Float32Bits(_) => self
  }
}

///|
/// Context closure for a single exact rule. No speculative reassociation and
/// no observation-dependent rewrite. The same rule is used by relation_to.
pub fn NumericExpression::simplify_exact(self : Self) -> Self {
  { ..self, root: self.root.exact_normal_form(self.inputs), }
}

///|
/// Conditional on the SAME declared input domain and primitive semantics.
/// Unknown rewrites are not disproven: they simply have no authorization here.
pub fn NumericExpression::relation_to(
  self : Self,
  candidate : Self,
) -> NumericRelation {
  if self.inputs != candidate.inputs || self.output != candidate.output {
    return DifferentProgram
  }
  if self.root == candidate.root {
    return IdenticalProgram
  }
  if self.root.exact_normal_form(self.inputs) ==
    candidate.root.exact_normal_form(candidate.inputs) {
    // Every finite BF16 value (including signed zero and subnormals) is
    // representable in F32. Exclude NaN payload/canonicalization assumptions.
    return ExactFiniteBf16Roundtrip
  }
  let rounding = match (self.root, candidate.root) {
    (Input(i), Convert(Float32, Convert(BFloat16, Input(j))))
    | (Convert(Float32, Convert(BFloat16, Input(j))), Input(i)) if i == j =>
      Some(i)
    _ => None
  }
  if rounding is Some(index) &&
    self.inputs[index] is (Float32, BoundedFinite(m)) {
    // BF16 RN error <= |x|/256 for normal results, <= 2^-134 in
    // the subnormal interval. Restrict to 2^126 to exclude overflow.
    // Use twice the relative bound + 2^-132: intentionally conservative,
    // including rounding of this F64 bound calculation. This is a LOCAL
    // single conversion theorem, never an accumulated model-error bound.
    if m <= 85070591730234615865843651857942052864.0 {
      return BoundedBf16Rounding(
        absolute_error=m / 128.0 + 1.8367099231598242e-40,
      )
    }
  }
  DifferentProgram
}

///|
/// Exact optimization gate, not an empirical "close enough" gate. The input
/// domains are preconditions the containing model/operator contract must own.
pub fn NumericExpression::exact_replacement(
  self : Self,
  candidate : Self,
) -> Self raise PrecisionPlanError {
  match self.relation_to(candidate) {
    IdenticalProgram | ExactFiniteBf16Roundtrip => candidate
    BoundedBf16Rounding(..) | DifferentProgram => raise MissingAccuracyContract
  }
}

///|
/// Explicit local approximation contract; never called by exact optimization.
/// No general composition theorem is claimed for a sequence of these rewrites.
pub fn NumericExpression::bounded_replacement(
  self : Self,
  candidate : Self,
  maximum_absolute_error~ : Double,
) -> Self raise PrecisionPlanError {
  guard maximum_absolute_error >= 0.0 &&
    !maximum_absolute_error.is_nan() &&
    !maximum_absolute_error.is_inf() else {
    raise MissingAccuracyContract
  }
  match self.relation_to(candidate) {
    IdenticalProgram | ExactFiniteBf16Roundtrip => candidate
    BoundedBf16Rounding(absolute_error~) if absolute_error <=
      maximum_absolute_error => candidate
    _ => raise MissingAccuracyContract
  }
}

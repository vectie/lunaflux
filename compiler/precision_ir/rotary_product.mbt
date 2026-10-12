///|
/// F32 adjacent complex product rounding before BF16 publication. This is a
/// numerical law, not an ISA or scheduling choice. For `(l,r)*(c,s)`, the
/// complex law evaluates real=fma(l,c,-round(r*s)) and
/// imag=fma(r,c,round(l*s)); inverse rotation conjugates `s` separately.
pub(all) enum RotaryProductLaw {
  SeparateFloat32Products
  ComplexFloat32Fma
} derive(Eq, Debug)

///|
pub extend RotaryProductLaw with Eq::{equal, not_equal}

///|
pub extend RotaryProductLaw with @debug.Debug::{to_repr}

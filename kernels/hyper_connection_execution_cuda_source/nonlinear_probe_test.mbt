///|
/// Offline arithmetic interventions only. The production renderer is not
/// modified; measured original mixes separate nonlinear error from reductions.
fn control_nonlinear_counterfactual(
  source : String,
  route : String,
) -> String raise {
  let text = if route.contains("direct") {
    let stable = "if (x >= 0.0f) return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); const float e = expf(x); return __fdiv_rn(e, __fadd_rn(1.0f, e));"
    assert_eq(source.split(stable).count(), 2)
    source.substitute(
      stable, "return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));",
    )
  } else {
    source
  }
  let text = if route.contains("affine_fma") {
    ["stream", "p", "control_index"].fold(init=text, (text, index) => {
      let scale = if index == "stream" {
        "0"
      } else if index == "p" {
        "1"
      } else {
        "2"
      }
      let separate = "__fadd_rn(__fmul_rn(mixes[" +
        index +
        "], scale[" +
        scale +
        "]), base[" +
        index +
        "])"
      assert_true(text.contains(separate))
      text.substitute(
        separate,
        "__fmaf_rn(mixes[" +
        index +
        "], scale[" +
        scale +
        "], base[" +
        index +
        "])",
      )
    })
  } else {
    text
  }
  if route == "refined_rsqrt" {
    let separate = "__fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(square_sum, 16384.0f), 0.000001f)))"
    assert_eq(text.split(separate).count(), 2)
    text.substitute(
      separate, "rsqrtf(__fadd_rn(__fdiv_rn(square_sum, 16384.0f), 0.000001f))",
    )
  } else {
    text
  }
}

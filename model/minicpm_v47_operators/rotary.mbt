///|
/// The selected model interleaves frequency axes, but pairs split the rotary
/// prefix in half. These are independent layouts. Keep model section policy
/// here; generic IR receives only its axis map and FP32 frequency constants.
fn full_rotary(
  spec : @model.Spec,
  heads : Int32,
  maximum_rows~ : Int32,
) -> @precision.MultiAxisRotaryPrecision raise {
  let rotary = spec.rotary()
  let width = spec.decoder().full_attention().head_width
  let prefix = (width.to_double() * rotary.partial_factor).to_int()
  let pairs = prefix / 2
  let axes = Array::makei(pairs, pair => {
    if pair % 3 == 1 && pair < rotary.height_sections * 3 {
      1
    } else if pair % 3 == 2 && pair < rotary.width_sections * 3 {
      2
    } else {
      0
    }
  })
  guard axes.filter(axis => axis == 0).length() == rotary.temporal_sections &&
    axes.filter(axis => axis == 1).length() == rotary.height_sections &&
    axes.filter(axis => axis == 2).length() == rotary.width_sections else {
    fail("inconsistent MiniCPM rotary sections")
  }
  let frequencies = Array::makei(pairs, pair => {
    // Upstream computes 1 / pow(base, arange(0,dim,2)/dim) in FP32.
    // Do not substitute pow(base,-exponent), which has a different round.
    let exponent = Float::from_int(pair * 2) / Float::from_int(prefix)
    1.0F / @math.powf(Float::from_double(rotary.theta), exponent)
  })
  @precision.MultiAxisRotaryPrecision::new(
    head_count=heads,
    head_width=width,
    rotary_width=prefix,
    maximum_rows~,
    maximum_positions=spec.decoder().context_length(),
    axis_for_pair=axes[:],
    inverse_frequencies=frequencies[:],
  )
}

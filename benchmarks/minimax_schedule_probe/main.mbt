///|
/// Compare the pure startup planner with actual original CPU outputs.
async fn main {
  guard @env.args() is [_, input] else { fail("ORIGINAL_SIGMA_TSV") }
  let text = @fs.read_file(input).text()
  let mut points = 0
  for line in text.split("\n") {
    if line.is_empty() || !line.contains("\t") {
      continue
    }
    guard line.split("\t").collect() is [count, shift, index, bits] else {
      fail("invalid original sigma row")
    }
    let count = @string.parse_int(count)
    let shift = @string.parse_int(shift)
    let index = @string.parse_int(index)
    let expected = @string.parse_uint(bits)
    let scheduler = @plan.JointDiffusionScheduler::new(
      latent=VideoLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=shift,
      shift_denominator=1,
      training_timestep_count=1000,
    )
    let actual = Float::from_double(
      scheduler.sigma_at(InclusiveZeroFloat32, count, index),
    ).reinterpret_as_uint()
    guard actual == expected else {
      fail(
        "sigma differs count=\{count} shift=\{shift} index=\{index} actual=\{actual} expected=\{expected}",
      )
    }
    points += 1
  }
  assert_eq(points, 2174)
  println(
    "original_sigma_comparison_complete points=\{points} bitwise_equal=true device=none",
  )
}

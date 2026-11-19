///|
test "BF16 rounding uses ties-even and IEEE special values without saturation" {
  let round = @bf16_operator_reference.round_bf16
  assert_eq(
    round(Float::reinterpret_from_uint(0x3f808000U)).reinterpret_as_uint(),
    0x3f800000U,
  )
  assert_eq(
    round(Float::reinterpret_from_uint(0x3f818000U)).reinterpret_as_uint(),
    0x3f820000U,
  )
  assert_eq(
    round(Float::reinterpret_from_uint(0x80000000U)).reinterpret_as_uint(),
    0x80000000U,
  )
  assert_eq(
    round(Float::reinterpret_from_uint(0x00008000U)).reinterpret_as_uint(),
    0U,
  )
  assert_eq(
    round(Float::reinterpret_from_uint(0x00018000U)).reinterpret_as_uint(),
    0x00020000U,
  )
  assert_true(round(Float::reinterpret_from_uint(0x7f7fffffU)).is_inf())
  assert_true(round(Float::reinterpret_from_uint(0xff800000U)).is_inf())
  assert_true(round(Float::reinterpret_from_uint(0x7f800001U)).is_nan())
}

///|
test "centered gamma is added in FP32 and strided views publish packed rows" {
  let plan = @precision.RmsNormPrecision::new(
    vector_width=1,
    vectors_per_row=1,
    maximum_rows=1,
    epsilon=1.0,
    mode=Weighted(OnePlusGamma),
  )
  let output = FixedArray::make(1, 0.0F)
  @bf16_operator_reference.rms_norm_into(
    plan,
    [1.0F],
    [0.00390625F],
    output,
    rows=1,
  )
  // Pre-rounding 1+gamma to BF16 would produce 0.70703125 instead.
  assert_eq(output[0], 0.7109375F)
  let views = @precision.RmsNormPrecision::new(
    vector_width=1,
    vectors_per_row=2,
    maximum_rows=2,
    epsilon=1.0,
    mode=Weighted(OnePlusGamma),
    input_vector_stride=2,
    input_row_stride=4L,
  )
  let packed = FixedArray::make(4, 0.0F)
  @bf16_operator_reference.rms_norm_into(
    views,
    [1.0F, 999.0F, 2.0F, 999.0F, 3.0F, 999.0F, 4.0F],
    [0.0F],
    packed,
    rows=2,
  )
  assert_eq(packed[:].to_owned(), [
    0.70703125F, 0.89453125F, 0.94921875F, 0.96875F,
  ])
}

///|
test "recurrent norm uses SiLU gate after BF16 normalized and weighted stages" {
  let plan = @precision.RmsNormPrecision::new(
    vector_width=1,
    vectors_per_row=1,
    maximum_rows=3,
    epsilon=1.0,
    mode=Bf16NormBf16WeightSiluGate,
  )
  let output = FixedArray::make(3, 0.0F)
  @bf16_operator_reference.rms_norm_into(
    plan,
    [1.0F, 1.0F, 1.0F],
    [1.5F],
    output,
    rows=3,
    gate=[1.0F, -1.0F, 2.0F],
  )
  // The third value would be 1.8671875 if the weighted BF16 round were skipped.
  assert_eq(output[:].to_owned(), [0.77734375F, -0.28515625F, 1.875F])
}

///|
test "routing ranks full softmax, rounds selected weights and reuses scratch" {
  let plan = @precision.SoftmaxTopKPrecision::new(
    expert_count=4,
    experts_per_token=2,
    maximum_rows=2,
  )
  let router = @bf16_operator_reference.Router::new(plan)
  let ids = FixedArray::make(4, -1)
  let weights = FixedArray::make(4, 0.0F)
  router.run_into(
    [0.0F, 1.0F, 3.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F],
    ids,
    weights,
    rows=2,
  )
  assert_eq(ids[:].to_owned(), [2, 3, 0, 1])
  assert_eq(weights[:].to_owned(), [0.73046875F, 0.26953125F, 0.5F, 0.5F])
  // Stable softmax at extreme logits; selected zero-probability experts stay unique.
  router.run_into(
    [-1000.0F, -1000.0F, 1000.0F, -1000.0F, 0.0F, 0.0F, 0.0F, 0.0F],
    ids,
    weights,
    rows=2,
  )
  assert_eq(ids[:].to_owned(), [2, 0, 0, 1])
  assert_eq(weights[:].to_owned(), [1.0F, 0.0F, 0.5F, 0.5F])
}

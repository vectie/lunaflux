///|
fn fp8_target_claims(
  major : Int,
  minor : Int,
  bf16? : Bool = true,
  cublas_lt? : Bool = true,
) -> I8RuntimeDescriptorExecutionClaims {
  {
    device_ordinal: 0,
    compute_major: major,
    compute_minor: minor,
    total_memory_bytes: 25769803776L,
    supports_bf16: bf16,
    supports_cublas_lt: cublas_lt,
    tokens_per_page: 16,
    total_page_count: 4096,
    model_generation: 1UL,
  }
}

///|
test "reusable FP8 descriptor admits exact sm120 software target" {
  let (capability, numeric, target) = admit_fp8_device_capability(
    fp8_target_claims(12, 0),
  )
  assert_eq(capability.compute_major(), 12)
  assert_eq(capability.compute_minor(), 0)
  assert_eq(target.compute_major(), 12)
  assert_eq(target.compute_minor(), 0)
  assert_eq(numeric.feature(), FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1)
}

///|
test "reusable FP8 descriptor rejects nonexact and incomplete targets" {
  for
    claims in [
      fp8_target_claims(10, 0),
      fp8_target_claims(12, 1),
      fp8_target_claims(12, 0, bf16=false),
      fp8_target_claims(12, 0, cublas_lt=false),
    ] {
    try admit_fp8_device_capability(claims) catch {
      AdmissionRejected(DeviceCapability) => ()
      _ => fail("invalid FP8 target reported an unrelated error")
    } noraise {
      _ => fail("invalid FP8 target entered descriptor admission")
    }
  }
}

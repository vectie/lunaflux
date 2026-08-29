///|
test "fused canary expectation is stable across repeated executions" {
  assert_eq(fused_expected_dispatch_canary(17U, 3), 51U)
  assert_eq(fused_expected_dispatch_canary(17U, 3), 51U)
  let encoded = FixedArray::make(4, b'\x00')
  encoded[0] = b'3'
  assert_eq(fused_decode_dispatch_canary(encoded), 51U)
}

///|
test "fused canary rejects guard return and wrong publication" {
  for observed in [0U, 50U, 52U] {
    try require_fused_dispatch_canary_observation(17U, 3, observed) catch {
      InvalidExecutor(ResourceState) => ()
      _ => fail("unexpected fused canary rejection")
    } noraise {
      _ => fail("wrong fused canary reached completion")
    }
  }
}

///|
test "fused canary rejects zero and multiplication overflow" {
  for case in 0..<3 {
    try {
      match case {
        0 => ignore(fused_expected_dispatch_canary(0U, 1))
        1 => ignore(fused_expected_dispatch_canary(1U, 0))
        _ => ignore(fused_expected_dispatch_canary(0xffffffffU, 2))
      }
    } catch {
      InvalidExecutor(ResourceState) => ()
      _ => fail("unexpected fused canary bounds rejection")
    } noraise {
      _ => fail("invalid fused canary bound reached execution")
    }
  }
}

///|
test "production residual RMSNorm ABI omits diagnostic argument" {
  assert_eq(fused_residual_argument_count_for_abi(ResidualRmsNormAbiV1), 7)
  assert_eq(
    fused_residual_argument_count_for_abi(ResidualRmsNormProductionAbiV2),
    6,
  )
  try
    ignore(
      fused_residual_argument_count_for_abi(QkvPositionedRopePagedKvWriteAbiV1),
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected foreign fused ABI rejection")
  } noraise {
    _ => fail("foreign fused ABI reached residual argument packing")
  }
}

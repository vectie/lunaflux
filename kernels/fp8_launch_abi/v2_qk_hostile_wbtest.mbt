///|
test "FP8 v2 fails closed for BF16-only Q/K RMSNorm" {
  try v2_operation_kind_tag(QkRmsNorm) catch {
    InvalidFp8LaunchAbi(OperationKind) => ()
    _ => fail("unexpected Q/K RMSNorm FP8 rejection")
  } noraise {
    _ => fail("Q/K RMSNorm must not gain an FP8 launch ABI implicitly")
  }
}

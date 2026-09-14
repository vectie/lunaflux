///|
test "fold observations require matched baseline unique roles and sample support" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let prefix = "luna-projection-fold-v1\nscope\tdevice\ttool\n"
  let baseline = "record\t\{a}\t\{a}\t100\t3\t-\n"
  for
    body in [
      "record\t\{a}\t\{b}\t50\t3\tmatrix:2:1:1\n",
      baseline + "record\t\{a}\t\{b}\t50\t2\tmatrix:2:1:1\n",
      baseline + "record\t\{a}\t\{b}\t50\t3\tmatrix:2:1:1,matrix:3:1:1\n",
      baseline + "record\t\{a}\t\{b}\t0\t3\tmatrix:2:1:1\n",
      baseline + baseline,
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(prefix + body),
        scope="device\ttool",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid observation accepted")
    }
  }
}

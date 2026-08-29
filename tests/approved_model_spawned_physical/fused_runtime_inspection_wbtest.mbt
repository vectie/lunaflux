///|
test "fused runtime inspection rejects duplicate and missing claims" {
  try fused_runtime_inspection_value("a=1\na=2\n", "a") catch {
    DeviceGreedyQualificationRejected => ()
    _ => abort("duplicate field raised the wrong error")
  } noraise {
    _ => abort("duplicate fused runtime field was accepted")
  }
  try fused_runtime_inspection_value("b=1\n", "a") catch {
    DeviceGreedyQualificationRejected => ()
    _ => abort("missing field raised the wrong error")
  } noraise {
    _ => abort("missing fused runtime field was accepted")
  }
}

///|
test "fused runtime inspection hex decoder is canonical" {
  assert_eq(fused_runtime_inspection_decode_hex("00617f"), b"\x00a\x7f")
  for hostile in ["", "0", "0A", "gg"] {
    try fused_runtime_inspection_decode_hex(hostile) catch {
      DeviceGreedyQualificationRejected => ()
      _ => abort("hostile hex raised the wrong error")
    } noraise {
      _ => abort("hostile fused runtime hex was accepted")
    }
  }
}

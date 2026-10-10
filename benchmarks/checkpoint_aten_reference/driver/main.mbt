///|
fn self_test(session : @native.Session, output : String) -> Unit raise {
  let x = session.run(Constant, a=2, b=4, d=2.0)
  defer x.close()
  let w = session.run(Constant, a=1, b=4, d=1.0)
  defer w.close()
  let normalized = session.run(RmsNorm, x~, y=w, d=0.000001)
  defer normalized.close()
  normalized.write(output + "/normalized.bf16")
  let bytes = session.bytes(output + "/normalized.bf16")
  guard bytes.length() == 16 else { fail("wrong BF16 output extent") }
  for i in 0..<8 {
    guard bytes[i * 2] == 0x80 && bytes[i * 2 + 1] == 0x3f else {
      fail("wrong independent RMSNorm result")
    }
  }
  try session.load(output + "/normalized.bf16", 0L, 4, 4) catch {
    _ => ()
  } noraise {
    t => {
      t.close()
      fail("short weight read must fail")
    }
  }
  try session.run(Constant, a=65536, b=65536) catch {
    _ => ()
  } noraise {
    t => {
      t.close()
      fail("memory budget must reject before allocation")
    }
  }
  guard session.bytes(output + "/normalized.bf16") == bytes else {
    fail("metadata read must recover after a failed native operation")
  }
  let recovered = session.run(Add, x~, y=x)
  defer recovered.close()
  normalized.close()
  recovered.close()
  w.close()
  x.close()
  x.close()
  guard session.retained_bytes() == 0L else {
    fail("failure or explicit cleanup leaked tensors")
  }
  try session.run(Add, x~, y=x) catch {
    _ => ()
  } noraise {
    t => {
      t.close()
      fail("closed operand must fail")
    }
  }
  println(
    "reference_self_test_passed arithmetic=bf16_rmsnorm short_read=clean budget=preallocation double_close=clean closed_operand=rejected retained_bytes=0",
  )
}

///|
fn main raise {
  match @env.args() {
    [_, "self-test", library, output] => {
      let session = @native.Session::new(library, device=0, budget=2097152L)
      defer session.close()
      self_test(session, output)
    }
    [_, "encode", library, model, text_ids, output, depth] => {
      let ids : Array[Int] = []
      for id in text_ids.split(",") {
        ids.push(@string.parse_int(id))
      }
      let layers = @string.parse_int(depth)
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      encode(session, model, ids, output, layers)
    }
    _ =>
      fail(
        "self-test LIB OUTPUT_DIRECTORY | encode LIB ORIGINAL_ENCODER_ROOT COMMA_IDS OUTPUT_DIRECTORY LAYERS",
      )
  }
}

///|
suberror TestSinkError {
  Stop(Int)
} derive(Debug, Eq)

///|
fn count_visit(
  visits : Ref[Int],
  _chunk : @materialize.TensorChunk,
) -> Unit raise TestSinkError {
  visits.val = visits.val + 1
  if visits.val < 0 {
    raise Stop(visits.val)
  }
}

///|
test "visitor receives exact bytes in semantic TensorRef order" {
  let bindings = test_bindings()
  let file = test_file(bindings.materialized_bytes().to_int())
  let references : Array[Int] = []
  let first_bytes : Array[Byte] = []
  let result = @materialize.visit(bindings, file, limits=test_limits(), chunk => {
    references.push(chunk.reference().as_int())
    guard chunk.bytes().get(0) is Some(first) else {
      fail("expected non-empty tensor")
    }
    first_bytes.push(first)
  })
  guard result is Ok(report) else {
    fail("expected valid materialization report")
  }
  assert_eq(references, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  assert_eq(first_bytes, [0, 64, 72, 104, 120, 136, 168, 176, 224, 21, 69, 77])
  assert_eq(report.length(), 12)
  assert_eq(report.identity(), bindings.identity())
  assert_eq(report.total_bytes(), bindings.materialized_bytes())
  let first_ref = try! @plan.TensorRef::new(0)
  guard report.receipt(first_ref) is Some(receipt) else {
    fail("expected first receipt")
  }
  assert_eq(receipt.reference(), first_ref)
  assert_eq(receipt.byte_length(), 64L)
  assert_eq(
    receipt.checksum().as_hex(),
    "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
  )
}

///|
test "host materialization copies exact tensor bytes under its own bound" {
  let bindings = test_bindings()
  let file = test_file(bindings.materialized_bytes().to_int())
  let source_before = file
  let host = @materialize.copy_to_host(bindings, file, limits=test_limits())
  assert_eq(host.length(), bindings.length())
  assert_eq(host.identity(), bindings.identity())
  assert_eq(host.total_bytes(), bindings.materialized_bytes())
  let first_ref = try! @plan.TensorRef::new(0)
  guard host.tensor(first_ref) is Some(tensor) else {
    fail("expected first host tensor")
  }
  assert_eq(tensor.reference(), first_ref)
  assert_eq(tensor.tensor().name(), "model.embed_tokens.weight")
  assert_eq(tensor.bytes(), Bytes::makei(64, index => index.to_byte()))
  assert_eq(
    tensor.checksum().as_hex(),
    "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
  )
  assert_eq(file, source_before)
}

///|
test "callback failure propagates unchanged and stops ordered delivery" {
  let bindings = test_bindings()
  let file = test_file(bindings.materialized_bytes().to_int())
  let visited : Array[Int] = []
  try
    @materialize.visit(bindings, file, limits=test_limits(), chunk => {
      let reference = chunk.reference().as_int()
      visited.push(reference)
      if reference == 2 {
        raise TestSinkError::Stop(reference)
      }
    })
  catch {
    TestSinkError::Stop(2) => ()
    _ => fail("unexpected visitor error")
  } noraise {
    _ => fail("expected visitor failure")
  }
  assert_eq(visited, [0, 1, 2])
}

///|
test "malformed envelope fails before visitor delivery" {
  let bindings = test_bindings()
  let visits = Ref(0)
  let visitor = chunk => count_visit(visits, chunk)
  match @materialize.visit(bindings, b"short", limits=test_limits(), visitor) {
    Err(MissingHeaderLength) => ()
    _ => fail("unexpected short-envelope result")
  }
  assert_eq(visits.val, 0)
  match
    @materialize.visit(
      bindings,
      Bytes::make(8, b'\xff'),
      limits=test_limits(),
      visitor,
    ) {
    Err(HeaderLengthOverflow) => ()
    _ => fail("unexpected header-overflow result")
  }
  assert_eq(visits.val, 0)
  let truncated = Bytes::makei(10, index => {
    if index == 0 {
      TEST_HEADER_BYTES.to_byte()
    } else {
      b'\x00'
    }
  })
  match @materialize.visit(bindings, truncated, limits=test_limits(), visitor) {
    Err(TruncatedHeader) => ()
    _ => fail("unexpected truncated-header result")
  }
  assert_eq(visits.val, 0)
  let short_payload = test_file(bindings.materialized_bytes().to_int() - 1)
  match
    @materialize.visit(bindings, short_payload, limits=test_limits(), visitor) {
    Err(EnvelopeLengthMismatch) => ()
    _ => fail("unexpected payload-mismatch result")
  }
  assert_eq(visits.val, 0)
}

///|
test "stream and host bounds fail closed" {
  let bindings = test_bindings()
  let file = test_file(bindings.materialized_bytes().to_int())
  match
    @materialize.visit(
      bindings,
      file,
      limits=test_limits(max_tensor_count=11),
      _ => (),
    ) {
    Err(TooManyTensors) => ()
    _ => fail("unexpected tensor-count result")
  }
  match
    @materialize.visit(
      bindings,
      file,
      limits=test_limits(max_payload_bytes=391L, max_host_bytes=391L),
      _ => (),
    ) {
    Err(PayloadTooLarge) => ()
    _ => fail("unexpected payload-bound result")
  }
  try
    @materialize.copy_to_host(
      bindings,
      file,
      limits=test_limits(max_host_bytes=391L),
    )
  catch {
    @materialize.MaterializationError::HostMaterializationTooLarge => ()
    _ => fail("unexpected host-bound error")
  } noraise {
    _ => fail("expected host-bound rejection")
  }
  try
    @materialize.MaterializationLimits::new(
      max_header_bytes=0,
      max_tensor_count=1,
      max_payload_bytes=1L,
      max_host_bytes=1L,
    )
  catch {
    InvalidLimit(HeaderBytes) => ()
    _ => fail("unexpected invalid-limit error")
  } noraise {
    _ => fail("expected invalid limit rejection")
  }
}

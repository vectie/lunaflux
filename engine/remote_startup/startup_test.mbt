///|
test "both ordinal-zero nodes retain exact local contracts in one deterministic group" {
  let plan = compiled_plan()
  let first = group(plan)
  let second = group(plan)
  assert_eq(first.digest_sha256(), second.digest_sha256())
  assert_true(first.rank(2) is None)
  for rank in 0..<2 {
    let contract = first.rank(rank).unwrap()
    assert_eq(contract.node().device_ordinal(), 0)
    assert_eq(contract.rank(), rank)
    assert_eq(contract.worker(), worker(plan, rank))
    assert_eq(contract.group_digest_sha256(), first.digest_sha256())
    assert_eq(contract.group_generation_value(), 1UL)
    assert_eq(contract.lease_millis(), 1000UL)
    let lease = contract.start_lease(500UL)
    assert_eq(lease.generation_value(), contract.group_generation_value())
    assert_eq(lease.deadline_millis(), 1500UL)
    assert_eq(lease.poll(1500UL).status(), Expired)
    assert_eq(contract.rendezvous().to_bytes(), rendezvous().to_bytes())
    let bytes = FixedArray::make(
      @remote_startup.remote_startup_frame_bytes() + 8,
      b'\xff',
    )
    let length = contract.copy_to(bytes)
    second.rank(rank).unwrap().verify_configure(bytes, length)
    assert_eq(bytes[length], b'\xff')
    // Snapshot does not borrow caller's transport storage.
    bytes[0] = 0
    let again = FixedArray::make(length, b'\x00')
    ignore(contract.copy_to(again))
    second.rank(rank).unwrap().verify_configure(again, length)
  }
}

///|
test "every byte including reserved padding and embedded worker data is bound" {
  let expected = group(compiled_plan()).rank(1).unwrap()
  let bytes = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  let length = expected.copy_to(bytes)
  for index in 0..<length {
    bytes[index] = bytes[index] ^ b'\x01'
    try expected.verify_configure(bytes, length) catch {
      Rejected(FrameMismatch) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("changed byte accepted at \{index}")
    }
    bytes[index] = bytes[index] ^ b'\x01'
  }
  expected.verify_configure(bytes, length)
}

///|
test "frames cannot cross ranks generations executables rendezvous leases or predecessors" {
  let plan = compiled_plan()
  let baseline = group(plan)
  let expected = baseline.rank(0).unwrap()
  let alternatives = [
    baseline.rank(1).unwrap(),
    group(plan, generation=2UL).rank(0).unwrap(),
    group(plan, nonce=b'\x02').rank(0).unwrap(),
    group(plan, lease=2000UL).rank(0).unwrap(),
    // A follower executable change also changes the leader's group binding.
    group(plan, follower_executable='a').rank(0).unwrap(),
    group(plan, predecessor=7UL).rank(0).unwrap(),
  ]
  for alternative in alternatives {
    let bytes = FixedArray::make(
      @remote_startup.remote_startup_frame_bytes(),
      b'\x00',
    )
    let length = alternative.copy_to(bytes)
    try expected.verify_configure(bytes, length) catch {
      Rejected(FrameMismatch) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("foreign frame accepted")
    }
  }
}

///|
test "invalid length cannot read or overwrite outside bounded frame" {
  let expected = group(compiled_plan()).rank(0).unwrap()
  let bytes = FixedArray::make(3, b'\x55')
  try expected.copy_to(bytes) catch {
    Rejected(Destination) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("short destination accepted")
  }
  assert_eq(bytes, [b'\x55', b'\x55', b'\x55'])
  for
    length in [
      -1,
      0,
      3,
      @remote_startup.remote_startup_frame_bytes(),
      2147483647,
    ] {
    try expected.verify_configure(bytes, length) catch {
      Rejected(FrameLength) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("bad length accepted")
    }
  }
}

///|
test "group construction rejects mismatched generation predecessor ordinal and limits" {
  let plan = compiled_plan()
  let cases = [
    (
      worker(plan, 1, generation=2UL),
      @remote_startup.RemoteStartupRule::Generation,
    ),
    (
      worker(plan, 1, predecessor=7UL),
      @remote_startup.RemoteStartupRule::Predecessor,
    ),
    (worker(plan, 1, ordinal=1), @remote_startup.RemoteStartupRule::Ordinal),
    (
      worker(plan, 1, input_tokens=32),
      @remote_startup.RemoteStartupRule::Limits,
    ),
  ]
  for (follower, rule) in cases {
    try
      @remote_startup.RemoteGroupStartup::new(
        plan,
        [worker(plan, 0), follower],
        [digest('e'), digest('f')],
        deployment_digest=digest('0'),
        group_generation=1UL,
        rendezvous=rendezvous(),
        lease_millis=1000UL,
      )
    catch {
      @remote_startup.Rejected(actual) => assert_eq(actual, rule)
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("mismatched group accepted")
    }
  }
}

///|
test "contract is exact and range arithmetic cannot wrap" {
  let device = @device.parse_contract(contract_text())
  device.check_range(0x7ffffffcUL, 4UL)
  device.check_range(0x80000000UL, 0UL)
  for
    (address, bytes) in [
      (0x7ffffffcUL, 8UL),
      (0xffffffffffffffffUL, 4UL),
      (0UL, 0xffffffffffffffffUL),
    ] {
    try device.check_range(address, bytes) catch {
      InvalidRange => ()
      _ => fail("wrong range error")
    } noraise {
      _ => fail("invalid range accepted")
    }
  }
  for
    text in [
      "{}",
      contract_text().replace(old="2147483648", new="4294967296"),
      contract_text().replace(old="65537", new="65536"),
      contract_text().replace(old="ordered-fp32", new="bf16"),
      contract_text().replace(old="{", new="{\"unknown\":true,"),
    ] {
    try @device.parse_contract(text) catch {
      ContractMismatch => ()
      _ => fail("wrong contract error")
    } noraise {
      _ => fail("incompatible contract accepted")
    }
  }
}

///|
test "actual LunaTile lowering preserves ordered operations and final HALT" {
  let device = @device.parse_contract(contract_text())
  let compiled = @compiler.lower(device, program(), bindings())
  assert_eq(compiled.chunk_count(), 1)
  let bytes = compiled.chunk(0).unwrap()
  assert_eq(bytes.length(), 384)
  for i, opcode in [14, 14, 2, 3, 15, 0] {
    assert_eq(bytes[i * 64].to_int(), opcode)
    for j in 44..<64 {
      assert_eq(bytes[i * 64 + j], b'\x00')
    }
  }
  assert_eq(bytes[32], b'\xf0')
  assert_eq(bytes[35], b'\x7f')
  assert_eq(bytes[2 * 64 + 4].to_int(), 8)
  assert_eq(bytes[4 * 64 + 8].to_int(), 8)
  assert_eq(bytes[4 * 64 + 32].to_int(), 4)
  assert_eq(export_fixture(contract_text()), export_fixture(contract_text()))
}

///|
test "backend rejects unsupported dtype operators loops and uninitialized scratch" {
  let device = @device.parse_contract(contract_text())
  try @compiler.lower(device, program(dtype=BFloat16), bindings()) catch {
    @compiler.UnsupportedTensor => ()
    _ => fail("dtype error")
  } noraise {
    _ => fail("BF16 silently coerced")
  }
  try @compiler.lower(device, program(kind=Maximum), bindings()) catch {
    @compiler.UnsupportedInstruction => ()
    _ => fail("operator error")
  } noraise {
    _ => fail("unsupported operation accepted")
  }
  try @compiler.lower(device, program(loops=[2L]), bindings()) catch {
    @compiler.UnsupportedProgram => ()
    _ => fail("loop error")
  } noraise {
    _ => fail("loop effects erased")
  }
  try @compiler.lower(device, program(initialized=false), bindings()) catch {
    @compiler.UninitializedRegister => ()
    _ => fail("initialization error")
  } noraise {
    _ => fail("uninitialized register accepted")
  }
}

///|
test "missing duplicate overlapping and out-of-range global bindings fail" {
  let device = @device.parse_contract(contract_text())
  let original = bindings()
  for
    bad in [
      [original[0], original[1]],
      [original[0], original[0], original[2]],
      [original[0], original[1], { tensor: id(3), address: 0x50000004UL, }],
      [original[0], original[1], { tensor: id(3), address: 0x7ffffffcUL, }],
    ] {
    let mut rejected = false
    ignore(@compiler.lower(device, program(), bad)) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}

///|
test "program splits at 2047 operations and rejects embedded HALT" {
  let op = @device.Instruction::copy(4096, 0, 4096)
  let compiled = @device.PreparedProgram::new(Array::make(2048, op))
  assert_eq(compiled.chunk_count(), 2)
  assert_eq(compiled.chunk(0).unwrap().length(), 2048 * 64)
  assert_eq(compiled.chunk(1).unwrap().length(), 2 * 64)
  for i in 0..<64 {
    assert_eq(compiled.chunk(0).unwrap()[2047 * 64 + i], b'\x00')
  }
  try
    @device.PreparedProgram::new([op, @device.Instruction::halt(), op])
  catch {
    InvalidProgram => ()
    _ => fail("wrong HALT rejection")
  } noraise {
    _ => fail("early HALT accepted")
  }
  try @device.Instruction::copy(1, 0, 4) catch {
    InvalidInstruction => ()
    _ => fail("wrong alias rejection")
  } noraise {
    _ => fail("partial scratch alias accepted")
  }
}

///|
test "launch receipt rejects stale completion and wraps sequence" {
  let owner = @device.LaunchOwner::new()
  assert_eq(owner.begin(0xffffffffU, 6), 0U)
  assert_false(owner.observe(0xffffffffU, 0xffffffffU, 5U, 0U, 6U))
  assert_false(owner.observe(0xffffffffU, 0U, 5U, 0U, 6U))
  assert_false(owner.observe(0U, 0U, 3U, 0U, 6U))
  assert_true(owner.observe(0U, 0U, 5U, 0U, 6U))
  assert_eq(owner.begin(0U, 3), 1U)
  try owner.observe(1U, 1U, 5U, 0U, 6U) catch {
    LaunchMismatch => ()
    _ => fail("wrong length rejection")
  } noraise {
    _ => fail("wrong program accepted")
  }
  try owner.begin(1U, 3) catch {
    InvalidState => ()
    _ => fail("wrong poisoned-owner error")
  } noraise {
    _ => fail("poisoned owner reused")
  }
}

///|
test "uncertain START and faults require a new owner after recovery" {
  for fault in [false, true] {
    let owner = @device.LaunchOwner::new()
    ignore(owner.begin(10U, 6))
    if fault {
      try owner.observe(11U, 11U, 9U, 1U, 6U) catch {
        DeviceFault => ()
        _ => fail("wrong device error")
      } noraise {
        _ => fail("device fault ignored")
      }
    } else {
      owner.invalidate()
    }
    try owner.observe(11U, 11U, 5U, 0U, 6U) catch {
      InvalidState => ()
      _ => fail("wrong invalidation error")
    } noraise {
      _ => fail("uncertain submission retried")
    }
  }
}

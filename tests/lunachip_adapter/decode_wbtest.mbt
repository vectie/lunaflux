///|
fn word(bytes : Bytes, index : Int32) -> UInt32 {
  let i = index * 4
  bytes[i].to_uint() |
  (bytes[i + 1].to_uint() << 8) |
  (bytes[i + 2].to_uint() << 16) |
  (bytes[i + 3].to_uint() << 24)
}

///|
test "resident packing preserves instructions across a former HALT and a 2048-slot window" {
  let first = @device.PreparedProgram::new(
    Array::make(2046, @device.Instruction::copy(1, 0, 1))[:],
  )
  let second = @device.PreparedProgram::new(
    Array::make(2, @device.Instruction::copy(2, 1, 1))[:],
  )
  let joined = @device.PreparedProgram::concatenate([first, second])
  assert_eq(joined.chunk_count(), 2)
  let full = joined.chunk(0).unwrap()
  let tail = joined.chunk(1).unwrap()
  assert_eq(full.length(), 2048 * 64)
  assert_eq(tail.length(), 2 * 64)
  assert_eq(word(full, 2045 * 16 + 1), 1U)
  assert_eq(word(full, 2046 * 16 + 1), 2U)
  assert_eq(word(full, 2047 * 16), 0U)
  assert_eq(word(tail, 1), 2U)
  assert_eq(word(tail, 16), 0U)
}

///|
test "resident position buffers are disjoint and packing reflects shared KV loads" {
  let device = @device.parse_contract(contract_text())
  let layout = @layout.prepare(device, metadata(), context=3)
  let a = @transfer.prepare_decode_step(device, layout, 0, resident=true)
  let b = @transfer.prepare_decode_step(device, layout, 1, resident=true)
  assert_eq(a.program_count(), 1)
  assert_eq(a.program(0).unwrap().chunk_count(), 4)
  assert_eq(b.program(0).unwrap().chunk_count(), 5)
  assert_eq(b.logits_address() - a.logits_address(), 607744UL)
  assert_eq(b.constants_address() - a.constants_address(), 1280UL)
  assert_true(a.constants_address() >= a.logits_address() + 3UL * 607744UL)
  assert_true(
    b.constants_address() + 1280UL <=
    layout.reserve().address + layout.reserve().bytes,
  )
}

///|
test "attention tile fits entire GQA pairs including the final partial group" {
  let device = @device.parse_contract(contract_text())
  let layout = @layout.prepare(device, metadata())
  for
    (context, expected) in [
      (1, [16]),
      (256, [16]),
      (384, [10, 6]),
      (2048, Array::make(8, 2)),
    ] {
    let layer = @transfer.prepare_decode_step(device, layout, context - 1)
      .program(0)
      .unwrap()
    let heads = []
    for c in 0..<layer.chunk_count() {
      let bytes = layer.chunk(c).unwrap()
      for i in 0..<(bytes.length() / 64) {
        let row = i * 16
        if word(bytes, row) == 11U && word(bytes, row + 1) == 12032U {
          let count = word(bytes, row + 4).reinterpret_as_int()
          assert_true(count <= 4096 && count % context == 0)
          heads.push(count / context)
        }
      }
    }
    assert_eq(heads, expected)
  }
}

///|
test "resident BF16 embedding reads one pointer at layer zero" {
  let device = @device.parse_contract(contract_text())
  let layout = @layout.prepare(device, metadata(), context=1)
  let direct = @transfer.prepare_decode_step(device, layout, 0, resident=true)
  let indirect = @transfer.prepare_decode_step(
    device,
    layout,
    0,
    resident=true,
    indirect_bf16=true,
  )
  assert_false(direct.uses_indirect_bf16())
  assert_true(indirect.uses_indirect_bf16())
  let a = direct.program(0).unwrap().chunk(0).unwrap()
  let b = indirect.program(0).unwrap().chunk(0).unwrap()
  assert_eq(word(b, 0), 14U)
  assert_eq(word(b, 4), 1024U)
  assert_eq(word(b, 8).to_uint64(), indirect.embedding_address())
  for i in 0..<(b.length() / 4) {
    if i == 10 {
      assert_eq(word(b, i), 1U)
      assert_eq(word(a, i), 0U)
    } else {
      assert_eq(word(b, i), word(a, i))
    }
  }
  for args in [(3UL, 64), (0UL, 63), (0x80000000UL, 64)] {
    try
      @device.Instruction::load_indirect_bf16(device, 0, args.0, args.1)
    catch {
      InvalidRange => ()
      _ => fail("unexpected indirect range error")
    } noraise {
      _ => fail("invalid pointer slot or count")
    }
  }
}

///|
test "transformer descriptor constructors reject alias and wide-stride hazards" {
  assert_eq(@device.Instruction::reduce(Maximum, 9, 0, 4096).words()[0], 8U)
  assert_eq(@device.Instruction::unary(Exp, 0, 0, 4096).words()[0], 11U)
  assert_eq(
    @device.Instruction::scalar(Add, 0, 0, Immediate(-1.0), 4096).words()[9],
    0xbf800000U,
  )
  for
    f in [
      () => @device.Instruction::scalar(Multiply, 0, 0, Broadcast(1), 2),
      () => @device.Instruction::unary(Exp, 1, 0, 2),
      () => @device.Instruction::silu_numerator(1, 0, 10, 2),
      () => @device.Instruction::scratch_gemv(0, 0, 128, 1, 64, 1, 64),
      () => @device.Instruction::scratch_gemv(256, 0, 128, 1, 256, 1, 256),
      () => {
        @device.Instruction::scratch_gemv(
          4096, 0, 16384, 4096, 4096, 32767, 32767,
        )
      },
      () => @device.Instruction::reduce(Sum, 32768, 0, 1),
    ] {
    try f() catch {
      InvalidInstruction => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("invalid transformer operand admitted")
    }
  }
}

///|
test "full decode covers all layers tied head and cross-launch scratch dependencies" {
  let device = @device.parse_contract(contract_text())
  let layout = @layout.prepare(device, metadata())
  for position in [0, 64] {
    let step = @transfer.prepare_decode_step(device, layout, position)
    assert_eq(step.program_count(), 29)
    let initialized = Array::make(32768, false)
    let mut projections = 0
    let mut combined_qkv = 0
    let mut logits_rows = 0U
    let mut kv_stores = 0
    let mut chunks = 0
    fn require(base : Int32, n : Int32) -> Unit raise {
      assert_true(base >= 0 && n > 0 && base + n <= 32768)
      for i in base..<(base + n) {
        assert_true(initialized[i], msg="uninitialized scratch \{i}")
      }
    }
    for layer in 0..<29 {
      let program = step.program(layer).unwrap()
      let kv_reads : Map[UInt64, Int32] = Map([])
      for c in 0..<program.chunk_count() {
        chunks += 1
        let bytes = program.chunk(c).unwrap()
        assert_true(bytes.length() <= 2048 * 64 && bytes.length() % 64 == 0)
        for p in 0..<(bytes.length() / 64) {
          let row = p * 16
          let op = word(bytes, row)
          let dst = word(bytes, row + 1).reinterpret_as_int()
          let a = word(bytes, row + 2).reinterpret_as_int()
          let b = word(bytes, row + 3).reinterpret_as_int()
          let n = word(bytes, row + 4).reinterpret_as_int()
          let k = word(bytes, row + 5).reinterpret_as_int()
          let addr = word(bytes, row + 8).to_uint64()
          if layer < 28 && op == 14U {
            for value in [false, true] {
              let first = layout.kv_head(layer, value, 0, 0).address
              let next = first + (position + 1).to_uint64() * 4096UL
              if addr >= first && addr < next {
                assert_eq(n, 128)
                kv_reads[addr] = kv_reads.get(addr).unwrap_or(0) + 1
              }
            }
          }
          if op == 0U {
            assert_eq(p, bytes.length() / 64 - 1)
            for j in 0..<16 {
              assert_eq(word(bytes, row + j), 0U)
            }
            continue
          }
          assert_true(p < bytes.length() / 64 - 1)
          for j in 11..<16 {
            assert_eq(word(bytes, row + j), 0U)
          }
          if op == 4U || op == 5U {
            require(a, k)
          } else if op != 14U {
            require(a, n)
          }
          if op == 2U || op == 3U || op == 13U {
            match word(bytes, row + 10) {
              0U => require(b, n)
              1U => require(b, 1)
              2U => ()
              _ => fail("invalid flags")
            }
          }
          if op == 5U {
            let bs = word(bytes, row + 6).reinterpret_as_int()
            let br = word(bytes, row + 7).reinterpret_as_int()
            for r in 0..<n {
              for col in 0..<k {
                require(b + r * br + col * bs, 1)
              }
            }
          }
          if op == 4U {
            projections += 1
            if layer < 28 && dst == 2048 {
              assert_eq((a, n, k), (1024, 4096, 1024))
              for (offset, first_row) in [(0, 0), (1, 2048), (2, 3072)] {
                let weight = layout.weight(2 + 11 * layer + offset).unwrap()
                assert_eq(addr + first_row.to_uint64() * 2048UL, weight.address)
              }
              combined_qkv += 1
            }
            if layer == 28 {
              assert_eq(
                addr,
                layout.lm_head().address + logits_rows.to_uint64() * 2048UL,
              )
              assert_eq(k, 1024)
              logits_rows += n.reinterpret_as_uint()
            }
          }
          if op == 15U {
            if layer < 28 {
              assert_eq(n, 1024)
              assert_eq(
                addr,
                layout.kv_head(layer, kv_stores % 2 == 1, position, 0).address,
              )
              kv_stores += 1
            } else {
              assert_eq(
                addr + n.to_uint64() * 4UL,
                step.logits_address() + logits_rows.to_uint64() * 4UL,
              )
            }
          } else {
            let count = if op >= 6U && op <= 8U { 1 } else { n }
            for i in dst..<(dst + count) {
              initialized[i] = true
            }
          }
        }
      }
      if layer < 28 {
        assert_eq(kv_reads.length(), position * 16)
        for p in 0..<position {
          for head in 0..<8 {
            for value in [false, true] {
              assert_eq(
                kv_reads.get(layout.kv_head(layer, value, p, head).address),
                Some(1),
              )
            }
          }
        }
      }
    }
    assert_eq(combined_qkv, 28)
    assert_eq(projections, 28 * 5 + 38)
    assert_eq(logits_rows, 151936U)
    assert_eq(kv_stores, 56)
    assert_true(chunks >= 29)
    if position == 64 {
      assert_true(chunks > 29)
    }
  }
}

///|
test "full-window decode and RoPE constants obey context and workspace bounds" {
  let device = @device.parse_contract(contract_text())
  let layout = @layout.prepare(device, metadata())
  let step = @transfer.prepare_decode_step(device, layout, 2047)
  assert_eq(step.program_count(), 29)
  assert_eq(step.program(0).unwrap().chunk_count(), 49)
  assert_eq(step.constants().length(), 1280)
  let zero = @transfer.prepare_decode_step(device, layout, 0).constants()
  for i in 0..<64 {
    assert_eq(word(zero, i), 0x3f800000U)
  }
  for i in 64..<128 {
    assert_eq(word(zero, i), 0U)
  }
  for i in 128..<192 {
    assert_eq(word(zero, i), 0x80000000U)
  }
  for i in 192..<320 {
    assert_eq(word(zero, i), 0U)
  }
  for p in [-1, 2048] {
    try @transfer.prepare_decode_step(device, layout, p) catch {
      InvalidPosition => ()
      _ => fail("wrong context error")
    } noraise {
      _ => fail("out-of-window position admitted")
    }
  }
}

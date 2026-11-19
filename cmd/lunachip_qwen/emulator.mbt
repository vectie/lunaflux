///|
priv struct Emulator {
  scratch : FixedArray[Float]
  ram : Map[UInt64, Float]
  source : Source
}

///|
fn Emulator::new(source : Source) -> Self raise {
  let ram : Map[UInt64, Float] = Map([])
  for transfer in source.transfers {
    if transfer.expanded_norm is Some(region) {
      let bytes = source.weight(transfer.ordinal)
      for i in 0..<(bytes.length() / 2) {
        ram[region.address + i.to_uint64() * 4UL] = bf16(bytes, i)
      }
    }
  }
  {
    scratch: FixedArray::make(32768, Float::reinterpret_from_uint(0x7fc00000U)),
    ram,
    source,
  }
}

///|
fn Emulator::write(self : Self, address : UInt64, bytes : Bytes) -> Unit {
  for i in 0..<(bytes.length() / 4) {
    self.ram[address + i.to_uint64() * 4UL] = Float::reinterpret_from_uint(
      decode_word(bytes, i),
    )
  }
}

///|
fn Emulator::read(self : Self, address : UInt64) -> Float raise {
  guard self.ram.get(address) is Some(value) else {
    fail("read of uninitialized DDR \{address}")
  }
  value
}

///|
fn Emulator::execute(
  self : Self,
  program : @device.PreparedProgram,
) -> Unit raise {
  let s = self.scratch
  for chunk in 0..<program.chunk_count() {
    let bytes = program.chunk(chunk).unwrap()
    for pc in 0..<(bytes.length() / 64) {
      let row = pc * 16
      let op = decode_word(bytes, row)
      if op == 0U {
        assert_eq(pc, bytes.length() / 64 - 1)
        break
      }
      let dst = decode_word(bytes, row + 1).to_int()
      let a = decode_word(bytes, row + 2).to_int()
      let b = decode_word(bytes, row + 3).to_int()
      let n = decode_word(bytes, row + 4).to_int()
      let k = decode_word(bytes, row + 5).to_int()
      let bs = decode_word(bytes, row + 6).to_int()
      let br = decode_word(bytes, row + 7).to_int()
      let address = decode_word(bytes, row + 8).to_uint64()
      let immediate = Float::reinterpret_from_uint(decode_word(bytes, row + 9))
      let flags = decode_word(bytes, row + 10)
      match op {
        4U | 5U => {
          let matrix = if op == 4U {
            self.source.matrix(address, n * k * 2)
          } else {
            b""
          }
          for r in 0..<n {
            let mut sum : Float = 0.0
            for col in 0..<k {
              let weight = if op == 4U {
                bf16(matrix, r * k + col)
              } else {
                s[b + r * br + col * bs]
              }
              let product = Float::from_double(
                s[a + col].to_double() * weight.to_double(),
              )
              sum = Float::from_double(sum.to_double() + product.to_double())
            }
            s[dst + r] = sum
          }
        }
        6U | 7U | 8U => {
          let mut sum : Float = if op == 8U {
            Float::reinterpret_from_uint(0xff800000U)
          } else {
            0.0
          }
          for i in 0..<n {
            let x = s[a + i]
            if op == 8U {
              if x > sum {
                sum = x
              }
            } else {
              let v = if op == 7U { x * x } else { x }
              sum = sum + v
            }
          }
          s[dst] = sum
        }
        14U =>
          if flags == 1U {
            let pointer = self.read(address).reinterpret_as_uint().to_uint64()
            guard n % 64 == 0 &&
              pointer % 128UL == 0UL &&
              pointer + n.to_uint64() * 2UL <= 0x80000000UL else {
              fail("invalid indirect BF16 span")
            }
            let row = self.source.matrix(pointer, n * 2)
            for i in 0..<n {
              s[dst + i] = bf16(row, i)
            }
          } else {
            for i in 0..<n {
              s[dst + i] = self.read(address + i.to_uint64() * 4UL)
            }
          }
        15U =>
          for i in 0..<n {
            self.ram[address + i.to_uint64() * 4UL] = s[a + i]
          }
        _ =>
          for i in 0..<n {
            let x = s[a + i]
            let y = match flags {
              0U => s[b + i]
              1U => s[b]
              _ => immediate
            }
            s[dst + i] = match op {
              1U => x
              2U => x + y
              3U => x * y
              9U | 10U | 11U => approximate(op, x)
              12U =>
                Float::reinterpret_from_uint(
                  x.reinterpret_as_uint() | 0x80000000U,
                )
              13U =>
                if (x.reinterpret_as_uint() & 0x80000000U) != 0U {
                  x * y
                } else {
                  x
                }
              _ => fail("unsupported opcode")
            }
          }
      }
    }
  }
}

///|
fn approximate(op : UInt32, x : Float) -> Float {
  if op == 11U {
    if x < -32.0 {
      return 0.0
    }
    if x > 32.0 {
      return Float::reinterpret_from_uint(0x7f800000U)
    }
    let reduced : Float = x * (1.0 / 64.0)
    let mut y = Float::from_double(1.0 / 40320.0)
    for
      c in [
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
      ] {
      let product = y * reduced
      y = product + Float::from_double(c)
    }
    for _ in 0..<6 {
      y = y * y
    }
    y
  } else if op == 9U {
    let mut y = Float::reinterpret_from_uint(
      0x7ef311c3U - x.reinterpret_as_uint(),
    )
    for _ in 0..<4 {
      let product = x * y
      let difference : Float = 2.0 - product
      y = y * difference
    }
    y
  } else {
    let mut y = Float::reinterpret_from_uint(
      0x5f375a86U - (x.reinterpret_as_uint() >> 1),
    )
    let half : Float = x * 0.5
    for _ in 0..<4 {
      let squared = y * y
      let product = half * squared
      let difference : Float = 1.5 - product
      y = y * difference
    }
    y
  }
}

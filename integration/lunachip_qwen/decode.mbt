///|
pub(all) suberror DecodeError {
  InvalidPosition
  InvalidLayout
  Device(@device.AdapterError)
}

///|
/// Diagnostic single-sequence step. Programs must execute in order with retained
/// scratch and exclusive device ownership. KV positions 0..<position must already
/// be committed for this sequence. Preparation is outside the execution loop.
pub struct DecodeStep {
  priv position : Int32
  priv programs : ReadOnlyArray[@device.PreparedProgram]
  priv embedding : UInt64
  priv indirect_bf16 : Bool
  priv constants : UInt64
  priv logits : UInt64
}

///|
pub fn DecodeStep::position(self : Self) -> Int32 {
  self.position
}

///|
pub fn DecodeStep::program_count(self : Self) -> Int32 {
  self.programs.length()
}

///|
pub fn DecodeStep::program(
  self : Self,
  index : Int32,
) -> @device.PreparedProgram? {
  self.programs.get(index)
}

///|
pub fn DecodeStep::embedding_address(self : Self) -> UInt64 {
  self.embedding
}

///|
pub fn DecodeStep::uses_indirect_bf16(self : Self) -> Bool {
  self.indirect_bf16
}

///|
pub fn DecodeStep::constants_address(self : Self) -> UInt64 {
  self.constants
}

///|
pub fn DecodeStep::logits_address(self : Self) -> UInt64 {
  self.logits
}

///|
/// FP32 cos[64], sin[64], -sin[64], then 128 positive zero words.
/// Qwen3 default RoPE: theta=1e6, head_dim=128, rotate_half convention.
pub fn DecodeStep::constants(self : Self) -> Bytes {
  let words : Array[Float] = []
  for kind in 0..<3 {
    for i in 0..<64 {
      let angle = self.position.to_double() /
        1000000.0.pow(i.to_double() / 64.0)
      let value = if kind == 0 { @math.cos(angle) } else { @math.sin(angle) }
      words.push(Float::from_double(if kind == 2 { -value } else { value }))
    }
  }
  for _ in 0..<128 {
    words.push(0.0)
  }
  let bytes = Array::make(words.length() * 4, b'\x00')
  for i, word in words {
    for k in 0..<4 {
      bytes[i * 4 + k] = (word.reinterpret_as_uint() >> (k * 8)).to_byte()
    }
  }
  Bytes::from_array(bytes[:])
}

///|
fn required_norm(
  layout : @layout.Layout,
  ordinal : Int32,
) -> UInt64 raise DecodeError {
  guard layout.expanded_norm(ordinal) is Some(region) else {
    raise InvalidLayout
  }
  region.address
}

///|
fn required_weight(
  layout : @layout.Layout,
  ordinal : Int32,
) -> UInt64 raise DecodeError {
  guard layout.weight(ordinal) is Some(region) else { raise InvalidLayout }
  region.address
}

///|
/// All 28 decoder layers, final RMSNorm and all 151936 tied output rows.
/// BF16 weights remain in DDR; activations and KV are FP32. This uses the
/// published approximate-unary numerical contract, not BF16 activation rounding.
pub fn prepare_decode_step(
  device : @device.DeviceContract,
  layout : @layout.Layout,
  position : Int32,
  resident~ : Bool = false,
  indirect_bf16~ : Bool = false,
) -> DecodeStep raise DecodeError {
  if position < 0 || position >= layout.context() {
    raise InvalidPosition
  }
  let base = layout.workspace().address
  if layout.workspace().bytes < 615936UL {
    raise InvalidLayout
  }
  let programs : Array[@device.PreparedProgram] = []
  let reserve = layout.reserve()
  let positions = layout.context().to_uint64()
  if resident && 1048576UL + positions * (607744UL + 1280UL) > reserve.bytes {
    raise InvalidLayout
  }
  // Keep per-position numerical evidence intact until the trial's timer stops.
  // The first MiB of reserve belongs to board qualification.
  let logits = if resident {
    reserve.address + 1048576UL + position.to_uint64() * 607744UL
  } else {
    base + 8192UL
  }
  let constants = if resident {
    reserve.address +
    1048576UL +
    positions * 607744UL +
    position.to_uint64() * 1280UL
  } else {
    base + 4096UL
  }
  for layer in 0..<28 {
    programs.push(
      prepare_decoder_layer(
        device,
        layout,
        layer,
        position,
        base,
        constants,
        indirect_bf16~,
      ),
    )
  }
  let operations : Array[@device.Instruction] = []
  let head = required_weight(layout, 0)
  let gamma = required_norm(layout, 309)
  try {
    operations.push(@device.Instruction::load(device, 10240, gamma, 1024))
    norm(operations, 1024, 0, 10240, 1024)
    let mut row = 0
    while row < 151936 {
      let rows = (151936 - row).min(4096)
      operations.push(
        @device.Instruction::bf16_gemv(
          device,
          16384,
          1024,
          head + row.to_uint64() * 2048UL,
          rows,
          1024,
        ),
      )
      operations.push(
        @device.Instruction::store(
          device,
          16384,
          logits + row.to_uint64() * 4UL,
          rows,
        ),
      )
      row += rows
    }
    programs.push(@device.PreparedProgram::new(operations[:]))
  } catch {
    e => raise Device(e)
  }
  {
    position,
    programs: if resident {
      ReadOnlyArray::from_array([
        @device.PreparedProgram::concatenate(programs[:]) catch {
          e => raise Device(e)
        },
      ])
    } else {
      ReadOnlyArray::from_array(programs)
    },
    embedding: base,
    indirect_bf16,
    constants,
    logits,
  }
}

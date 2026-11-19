///|
fn transformer_unary(op : Int32, x : Float) -> Float {
  if op == 11 {
    let z : Float = x / 64.0
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
      let p = Float::from_double(y.to_double() * z.to_double())
      y = Float::from_double(p.to_double() + Float::from_double(c).to_double())
    }
    for _ in 0..<6 {
      y = Float::from_double(y.to_double() * y.to_double())
    }
    y
  } else if op == 9 {
    let mut y = Float::reinterpret_from_uint(
      0x7ef311c3U - x.reinterpret_as_uint(),
    )
    for _ in 0..<4 {
      let product = Float::from_double(x.to_double() * y.to_double())
      let difference = Float::from_double(2.0 - product.to_double())
      y = Float::from_double(y.to_double() * difference.to_double())
    }
    y
  } else {
    let mut y = Float::reinterpret_from_uint(
      0x5f375a86U - (x.reinterpret_as_uint() >> 1),
    )
    let half : Float = x * 0.5
    for _ in 0..<4 {
      let square = Float::from_double(y.to_double() * y.to_double())
      let product = Float::from_double(half.to_double() * square.to_double())
      let difference = Float::from_double(1.5 - product.to_double())
      y = Float::from_double(y.to_double() * difference.to_double())
    }
    y
  }
}

///|
/// Cross-repository ABI fixture: reductions, softmax, stable SiLU and a strided
/// scratch matrix. Expected arithmetic uses separate explicit binary32 rounding.
pub fn export_transformer_fixture(contract : String) -> String raise {
  let device = @device.parse_contract(contract)
  let x : Array[Float] = [2.0, 4.0, -1.0, 0.5]
  let matrix : Array[Float] = [1.0, 2.0, -2.0, 4.0, 3.0, -1.0, 0.5, 8.0]
  let ops = [
    @device.Instruction::load(device, 0, 0x70000000UL, 4),
    @device.Instruction::load(device, 16, 0x70000100UL, 8),
    @device.Instruction::scratch_gemv(64, 0, 16, 2, 4, 2, 1),
    @device.Instruction::reduce(Sum, 80, 0, 4),
    @device.Instruction::reduce(SumSquares, 81, 0, 4),
    @device.Instruction::reduce(Maximum, 82, 0, 4),
    @device.Instruction::unary(Reciprocal, 96, 82, 1),
    @device.Instruction::unary(ReciprocalSqrt, 97, 82, 1),
    @device.Instruction::unary(NegativeAbs, 100, 0, 4),
    @device.Instruction::unary(Exp, 104, 100, 4),
    @device.Instruction::scalar(Add, 108, 104, Immediate(1.0), 4),
    @device.Instruction::unary(Reciprocal, 108, 108, 4),
    @device.Instruction::silu_numerator(112, 0, 104, 4),
    @device.Instruction::binary(Multiply, 112, 112, 108, 4),
    @device.Instruction::scalar(Multiply, 83, 82, Immediate(-1.0), 1),
    @device.Instruction::scalar(Add, 120, 0, Broadcast(83), 4),
    @device.Instruction::unary(Exp, 120, 120, 4),
    @device.Instruction::reduce(Sum, 84, 120, 4),
    @device.Instruction::unary(Reciprocal, 84, 84, 1),
    @device.Instruction::scalar(Multiply, 120, 120, Broadcast(84), 4),
  ]
  let reciprocal = transformer_unary(9, 4.0)
  let rsqrt = transformer_unary(10, 4.0)
  let exponential = x.map(v => transformer_unary(11, -v.abs()))
  let silu = Array::makei(4, i => {
    let denominator : Float = 1.0 + exponential[i]
    let numerator = if x[i] < 0.0 { x[i] * exponential[i] } else { x[i] }
    numerator * transformer_unary(9, denominator)
  })
  let probability = x.map(v => transformer_unary(11, v - 4.0))
  let mut sum : Float = 0.0
  for value in probability {
    sum = sum + value
  }
  let inverse = transformer_unary(9, sum)
  for i in 0..<4 {
    probability[i] = probability[i] * inverse
  }
  let outputs : Array[Json] = []
  let mut address = 0x71000004UL
  for
    (base, values) in [
      (64, [-8.75F, 25.0F]),
      (80, [5.5F, 21.25F, 4.0F]),
      (96, [reciprocal, rsqrt]),
      (100, [-2.0F, -4.0F, -1.0F, -0.5F]),
      (104, exponential),
      (112, silu),
      (120, probability),
    ] {
    ops.push(@device.Instruction::store(device, base, address, values.length()))
    outputs.push({
      "address": address,
      "words": values.map(v => v.reinterpret_as_uint()),
    })
    address += 128UL
  }
  let program = @device.PreparedProgram::new(ops[:])
  let identity = @crypto.bytes_to_hex_string(
    @crypto.sha256(program.chunk(0).unwrap()),
  )
  let assets : Array[Json] = [
    { "address": "1879048192", "words": x.map(v => v.reinterpret_as_uint()) },
    {
      "address": "1879048448",
      "words": matrix.map(v => v.reinterpret_as_uint()),
    },
  ]
  let fixture : Json = {
    "schema": "lunachip.adapter-fixture/v1",
    "source_program": identity,
    "program_words": descriptor_words(program),
    "assets": assets,
    "outputs": outputs,
  }
  fixture.stringify(indent=2)
}

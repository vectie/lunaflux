///|
/// Separately expressed model equations. This reference never decodes device
/// instructions and uses libm exp/sqrt/division rather than the FPGA approximations.
priv struct Reference {
  source : Source
  keys : Array[Array[Array[Float]]]
  values : Array[Array[Array[Float]]]
}

///|
fn Reference::new(source : Source) -> Self {
  {
    source,
    keys: Array::makei(28, _ => []),
    values: Array::makei(28, _ => []),
  }
}

///|
fn reference_projection(
  source : Source,
  ordinal : Int32,
  input : ArrayView[Float],
) -> Array[Float] raise {
  let shape = source.layout.dimensions(ordinal).unwrap()
  let rows = shape[0].to_int()
  let columns = shape[1].to_int()
  assert_eq(input.length(), columns)
  let output = Array::make(rows, 0.0F)
  let mut first = 0
  while first < rows {
    let count = (rows - first).min(4096)
    let weights = source.weight(
      ordinal,
      offset=first * columns * 2,
      count=count * columns * 2,
    )
    for r in 0..<count {
      let mut accumulator : Float = 0.0
      for c in 0..<columns {
        let product = Float::from_double(
          bf16(weights, r * columns + c).to_double() * input[c].to_double(),
        )
        accumulator = Float::from_double(
          accumulator.to_double() + product.to_double(),
        )
      }
      output[first + r] = accumulator
    }
    first += count
  }
  output
}

///|
fn reference_norm(input : ArrayView[Float], weights : Bytes) -> Array[Float] {
  let mut sum : Float = 0.0
  for x in input {
    let square = x * x
    sum = sum + square
  }
  let mean = sum * Float::from_double(1.0 / input.length().to_double())
  let variance : Float = mean + 0.000001
  let inverse = Float::from_double(1.0 / variance.to_double().sqrt())
  Array::makei(input.length(), i => {
    let normalized = input[i] * inverse
    normalized * bf16(weights, i)
  })
}

///|
fn reference_heads(
  input : Array[Float],
  gamma : Bytes,
  position : Int32,
) -> Array[Float] {
  let result : Array[Float] = []
  for head in 0..<(input.length() / 128) {
    let normalized = reference_norm(input[head * 128:head * 128 + 128], gamma)
    let rotated = Array::make(128, 0.0F)
    for i in 0..<64 {
      let angle = position.to_double() *
        @math.pow(1000000.0, -i.to_double() / 64.0)
      let cos = Float::from_double(@math.cos(angle))
      let sin = Float::from_double(@math.sin(angle))
      let left = normalized[i] * cos
      let right = normalized[i + 64] * -sin
      rotated[i] = left + right
      let left = normalized[i + 64] * cos
      let right = normalized[i] * sin
      rotated[i + 64] = left + right
    }
    result.append(rotated)
  }
  result
}

///|
fn Reference::step(
  self : Self,
  token : Int32,
  position : Int32,
) -> Array[Float] raise {
  let mut x = self.source.embedding(token)
  for layer in 0..<28 {
    let ordinal = 1 + layer * 11
    let normalized = reference_norm(x[:], self.source.weight(ordinal))
    let query = reference_heads(
      reference_projection(self.source, ordinal + 1, normalized[:]),
      self.source.weight(ordinal + 4),
      position,
    )
    let key = reference_heads(
      reference_projection(self.source, ordinal + 2, normalized[:]),
      self.source.weight(ordinal + 5),
      position,
    )
    let value = reference_projection(self.source, ordinal + 3, normalized[:])
    assert_eq(self.keys[layer].length(), position)
    self.keys[layer].push(key)
    self.values[layer].push(value)
    let context = position + 1
    let attention = Array::make(2048, 0.0F)
    for h in 0..<16 {
      let scores = Array::make(context, 0.0F)
      let mut maximum = Float::reinterpret_from_uint(0xff800000U)
      for p in 0..<context {
        let mut sum : Float = 0.0
        for d in 0..<128 {
          let product = query[h * 128 + d] *
            self.keys[layer][p][h / 2 * 128 + d]
          sum = sum + product
        }
        scores[p] = sum * Float::from_double(1.0 / 128.0.sqrt())
        if scores[p] > maximum {
          maximum = scores[p]
        }
      }
      let mut denominator : Float = 0.0
      for p in 0..<context {
        let shifted = scores[p] - maximum
        scores[p] = Float::from_double(@math.exp(shifted.to_double()))
        denominator = denominator + scores[p]
      }
      let inverse : Float = 1.0 / denominator
      for p in 0..<context {
        scores[p] = scores[p] * inverse
      }
      for d in 0..<128 {
        let mut sum : Float = 0.0
        for p in 0..<context {
          let product = scores[p] * self.values[layer][p][h / 2 * 128 + d]
          sum = sum + product
        }
        attention[h * 128 + d] = sum
      }
    }
    let projected = reference_projection(self.source, ordinal + 6, attention[:])
    let residual = Array::makei(1024, i => x[i] + projected[i])
    let normalized = reference_norm(
      residual[:],
      self.source.weight(ordinal + 7),
    )
    let gate = reference_projection(self.source, ordinal + 8, normalized[:])
    let up = reference_projection(self.source, ordinal + 9, normalized[:])
    let activation = Array::makei(3072, i => {
      let a = gate[i].to_double()
      let sigmoid = 1.0 / (1.0 + @math.exp(-a))
      Float::from_double(a * sigmoid) * up[i]
    })
    let down = reference_projection(self.source, ordinal + 10, activation[:])
    x = Array::makei(1024, i => residual[i] + down[i])
  }
  let normalized = reference_norm(x[:], self.source.weight(309))
  reference_projection(self.source, 0, normalized[:])
}

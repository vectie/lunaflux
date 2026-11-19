///|
fn norm(
  ops : Array[@device.Instruction],
  dst : Int32,
  input : Int32,
  gamma : Int32,
  count : Int32,
) -> Unit raise @device.AdapterError {
  ops.push(@device.Instruction::reduce(SumSquares, 11904, input, count))
  ops.push(
    @device.Instruction::scalar(
      Multiply,
      11904,
      11904,
      Immediate(Float::from_double(1.0 / count.to_double())),
      1,
    ),
  )
  ops.push(
    @device.Instruction::scalar(Add, 11904, 11904, Immediate(0.000001), 1),
  )
  ops.push(@device.Instruction::unary(ReciprocalSqrt, 11904, 11904, 1))
  ops.push(
    @device.Instruction::scalar(Multiply, dst, input, Broadcast(11904), count),
  )
  ops.push(@device.Instruction::binary(Multiply, dst, dst, gamma, count))
}

///|
fn norm_query_key_heads(
  ops : Array[@device.Instruction],
) -> Unit raise @device.AdapterError {
  // Each sum still consumes one 128-element head in order. The 24 independent
  // scalar results then use the same three operations as an ordinary RMSNorm.
  let scales = 11904
  for h in 0..<24 {
    ops.push(
      @device.Instruction::reduce(SumSquares, scales + h, 2048 + h * 128, 128),
    )
  }
  ops.push(
    @device.Instruction::scalar(
      Multiply,
      scales,
      scales,
      Immediate(1.0 / 128.0),
      24,
    ),
  )
  ops.push(
    @device.Instruction::scalar(Add, scales, scales, Immediate(0.000001), 24),
  )
  ops.push(@device.Instruction::unary(ReciprocalSqrt, scales, scales, 24))
  for h in 0..<24 {
    let a = 2048 + h * 128
    let gamma = if h < 16 { 11264 } else { 11392 }
    ops.push(
      @device.Instruction::scalar(Multiply, a, a, Broadcast(scales + h), 128),
    )
    ops.push(@device.Instruction::binary(Multiply, a, a, gamma, 128))
  }
}

///|
// The largest historical-key tile ends at word 24576. These three tables
// survive all decoder layers and are refreshed at the start of each token.
const ROPE_COS : Int32 = 24576

///|
const ROPE_SIN : Int32 = 27648

///|
const ATTENTION_ZEROS : Int32 = 30720

///|
fn load_position_tables(
  device : @device.DeviceContract,
  ops : Array[@device.Instruction],
  constants : UInt64,
) -> Unit raise @device.AdapterError {
  ops.push(@device.Instruction::load(device, ROPE_COS, constants, 64))
  ops.push(@device.Instruction::copy(ROPE_COS + 64, ROPE_COS, 64))
  ops.push(@device.Instruction::load(device, ROPE_SIN, constants + 256UL, 128))
  ops.push(
    @device.Instruction::load(device, ATTENTION_ZEROS, constants + 768UL, 128),
  )
  for table in [ROPE_COS, ROPE_SIN] {
    for filled in [128, 256, 512, 1024, 2048] {
      ops.push(
        @device.Instruction::copy(
          table + filled,
          table,
          filled.min(3072 - filled),
        ),
      )
    }
  }
}

///|
fn rotate_queries_and_keys(
  ops : Array[@device.Instruction],
) -> Unit raise @device.AdapterError {
  // Sixteen query heads and eight key heads occupy consecutive scratch words.
  // The sine table holds +sin for the first half and -sin for the second.
  // Save both cross terms before replacing the input with the cosine product.
  ops.push(@device.Instruction::binary(Multiply, 6144, 2048, ROPE_SIN, 3072))
  ops.push(@device.Instruction::binary(Multiply, 2048, 2048, ROPE_COS, 3072))
  for h in 0..<24 {
    let a = 2048 + h * 128
    let cross = 6144 + h * 128
    ops.push(@device.Instruction::binary(Add, a + 64, a + 64, cross, 64))
    ops.push(@device.Instruction::binary(Add, a, a, cross + 64, 64))
  }
}

///|
fn kv_address(
  layout : @layout.Layout,
  layer : Int32,
  value : Bool,
  position : Int32,
  head : Int32,
) -> UInt64 raise DecodeError {
  (layout.kv_head(layer, value, position, head) catch {
    _ => raise InvalidLayout
  }).address
}

///|
fn prepare_decoder_layer(
  device : @device.DeviceContract,
  layout : @layout.Layout,
  layer : Int32,
  position : Int32,
  workspace : UInt64,
  constants : UInt64,
  indirect_bf16~ : Bool = false,
) -> @device.PreparedProgram raise DecodeError {
  let ordinal = 1 + layer * 11
  let input_gamma = required_norm(layout, ordinal)
  let q_gamma = required_norm(layout, ordinal + 4)
  let k_gamma = required_norm(layout, ordinal + 5)
  let post_gamma = required_norm(layout, ordinal + 7)
  let weights : Array[UInt64] = []
  for i in [1, 2, 3, 6, 8, 9, 10] {
    weights.push(required_weight(layout, ordinal + i))
  }
  let key = kv_address(layout, layer, false, position, 0)
  let value = kv_address(layout, layer, true, position, 0)
  let key_root = kv_address(layout, layer, false, 0, 0)
  let value_root = kv_address(layout, layer, true, 0, 0)
  let ops : Array[@device.Instruction] = []
  try {
    if layer == 0 {
      ops.push(
        if indirect_bf16 {
          @device.Instruction::load_indirect_bf16(device, 0, workspace, 1024)
        } else {
          @device.Instruction::load(device, 0, workspace, 1024)
        },
      )
      load_position_tables(device, ops, constants)
    }
    ops.push(@device.Instruction::load(device, 10240, input_gamma, 1024))
    norm(ops, 1024, 0, 10240, 1024)
    if weights[1] == weights[0] + 2048UL * 2048UL &&
      weights[2] == weights[1] + 1024UL * 2048UL {
      ops.push(
        @device.Instruction::bf16_gemv(
          device,
          2048,
          1024,
          weights[0],
          4096,
          1024,
        ),
      )
    } else {
      for (i, dst, rows) in [(0, 2048, 2048), (1, 4096, 1024), (2, 5120, 1024)] {
        ops.push(
          @device.Instruction::bf16_gemv(
            device,
            dst,
            1024,
            weights[i],
            rows,
            1024,
          ),
        )
      }
    }
    ops.push(@device.Instruction::load(device, 11264, q_gamma, 128))
    ops.push(@device.Instruction::load(device, 11392, k_gamma, 128))
    norm_query_key_heads(ops)
    rotate_queries_and_keys(ops)
    ops.push(@device.Instruction::store(device, 4096, key, 1024))
    ops.push(@device.Instruction::store(device, 5120, value, 1024))
    attention(device, ops, position + 1, key_root, value_root)
    ops.push(
      @device.Instruction::bf16_gemv(device, 1024, 6144, weights[3], 1024, 2048),
    )
    ops.push(@device.Instruction::binary(Add, 8192, 0, 1024, 1024))
    ops.push(@device.Instruction::load(device, 10240, post_gamma, 1024))
    norm(ops, 9216, 8192, 10240, 1024)
    ops.push(
      @device.Instruction::bf16_gemv(
        device,
        10240,
        9216,
        weights[4],
        3072,
        1024,
      ),
    )
    ops.push(
      @device.Instruction::bf16_gemv(
        device,
        13312,
        9216,
        weights[5],
        3072,
        1024,
      ),
    )
    ops.push(@device.Instruction::unary(NegativeAbs, 1024, 10240, 3072))
    ops.push(@device.Instruction::unary(Exp, 1024, 1024, 3072))
    ops.push(@device.Instruction::scalar(Add, 4096, 1024, Immediate(1.0), 3072))
    ops.push(@device.Instruction::unary(Reciprocal, 4096, 4096, 3072))
    ops.push(@device.Instruction::silu_numerator(1024, 10240, 1024, 3072))
    ops.push(@device.Instruction::binary(Multiply, 1024, 1024, 4096, 3072))
    ops.push(@device.Instruction::binary(Multiply, 10240, 1024, 13312, 3072))
    ops.push(
      @device.Instruction::bf16_gemv(
        device,
        9216,
        10240,
        weights[6],
        1024,
        3072,
      ),
    )
    ops.push(@device.Instruction::binary(Add, 0, 8192, 9216, 1024))
    @device.PreparedProgram::new(ops[:])
  } catch {
    e => raise Device(e)
  }
}

///|
// Keep complete pairs of query heads together in the 4096-word score area.
// Short contexts fit all sixteen heads. At the full window one pair fits.
fn attention(
  device : @device.DeviceContract,
  ops : Array[@device.Instruction],
  context : Int32,
  key : UInt64,
  value : UInt64,
) -> Unit raise @device.AdapterError {
  let pairs = (4096 / (2 * context)).min(8)
  let mut first_pair = 0
  while first_pair < 8 {
    let count = pairs.min(8 - first_pair)
    for i in 0..<count {
      attention_scores(
        device,
        ops,
        context,
        key,
        first_pair + i,
        12032 + 2 * i * context,
      )
    }
    attention_probabilities(ops, 12032, context, 2 * count)
    for i in 0..<count {
      attention_values(
        device,
        ops,
        context,
        value,
        first_pair + i,
        12032 + 2 * i * context,
      )
    }
    first_pair += count
  }
}

///|
fn attention_scores(
  device : @device.DeviceContract,
  ops : Array[@device.Instruction],
  context : Int32,
  key : UInt64,
  group : Int32,
  base : Int32,
) -> Unit raise @device.AdapterError {
  let history = context - 1
  let head = group.to_uint64() * 512UL
  let mut first = 0
  while first < history {
    let count = (history - first).min(64)
    for i in 0..<count {
      ops.push(
        @device.Instruction::load(
          device,
          16384 + i * 128,
          key + (first + i).to_uint64() * 4096UL + head,
          128,
        ),
      )
    }
    for q in 0..<2 {
      ops.push(
        @device.Instruction::scratch_gemv(
          base + q * context + first,
          2048 + (2 * group + q) * 128,
          16384,
          count,
          128,
          1,
          128,
        ),
      )
    }
    first += count
  }
  for q in 0..<2 {
    let scores = base + q * context
    ops.push(
      @device.Instruction::scratch_gemv(
        scores + history,
        2048 + (2 * group + q) * 128,
        4096 + group * 128,
        1,
        128,
        1,
        128,
      ),
    )
  }
}

///|
fn attention_values(
  device : @device.DeviceContract,
  ops : Array[@device.Instruction],
  context : Int32,
  value : UInt64,
  group : Int32,
  base : Int32,
) -> Unit raise @device.AdapterError {
  let history = context - 1
  let head = group.to_uint64() * 512UL
  for p in 0..<context {
    let source = if p == history {
      5120 + group * 128
    } else {
      ops.push(
        @device.Instruction::load(
          device,
          16384,
          value + p.to_uint64() * 4096UL + head,
          128,
        ),
      )
      16384
    }
    for q in 0..<2 {
      let dst = 6144 + (2 * group + q) * 128
      ops.push(
        @device.Instruction::scalar(
          Multiply,
          16512,
          source,
          Broadcast(base + q * context + p),
          128,
        ),
      )
      ops.push(
        @device.Instruction::binary(
          Add,
          dst,
          if p == 0 {
            ATTENTION_ZEROS
          } else {
            dst
          },
          16512,
          128,
        ),
      )
    }
  }
}

///|
fn attention_probabilities(
  ops : Array[@device.Instruction],
  scores : Int32,
  context : Int32,
  heads : Int32,
) -> Unit raise @device.AdapterError {
  let all_scores = context * heads
  let scales = 11904
  ops.push(
    @device.Instruction::scalar(
      Multiply,
      scores,
      scores,
      Immediate(Float::from_double(1.0 / 128.0.sqrt())),
      all_scores,
    ),
  )
  for h in 0..<heads {
    ops.push(
      @device.Instruction::reduce(
        Maximum,
        scales + h,
        scores + h * context,
        context,
      ),
    )
  }
  ops.push(
    @device.Instruction::scalar(
      Multiply,
      scales,
      scales,
      Immediate(-1.0),
      heads,
    ),
  )
  for h in 0..<heads {
    let row = scores + h * context
    ops.push(
      @device.Instruction::scalar(Add, row, row, Broadcast(scales + h), context),
    )
  }
  ops.push(@device.Instruction::unary(Exp, scores, scores, all_scores))
  for h in 0..<heads {
    ops.push(
      @device.Instruction::reduce(
        Sum,
        scales + h,
        scores + h * context,
        context,
      ),
    )
  }
  ops.push(@device.Instruction::unary(Reciprocal, scales, scales, heads))
  for h in 0..<heads {
    let row = scores + h * context
    ops.push(
      @device.Instruction::scalar(
        Multiply,
        row,
        row,
        Broadcast(scales + h),
        context,
      ),
    )
  }
}

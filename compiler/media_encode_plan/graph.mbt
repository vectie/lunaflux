///|
/// Family-neutral F32 channel-major encoder operations. Geometry is static;
/// host effects and device lowering are deliberately absent from this IR.
pub(all) enum Operation {
  Pad(batch~ : Int, input_time~ : Int, output_time~ : Int)
  WeightNorm(outputs~ : Int, per_output~ : Int)
  Conv(
    batch~ : Int,
    input_channels~ : Int,
    output_channels~ : Int,
    input_time~ : Int,
    output_time~ : Int,
    kernel~ : Int,
    dilation~ : Int,
    stride~ : Int,
    padding~ : Int
  )
  Snake(batch~ : Int, channels~ : Int, time~ : Int)
  Add(elements~ : Int64)
  LayerNorm(batch~ : Int, channels~ : Int, time~ : Int)
  Dense(
    batch~ : Int,
    input_channels~ : Int,
    output_channels~ : Int,
    time~ : Int
  )
  Qkv(batch~ : Int, channels~ : Int, time~ : Int)
  CausalAttention(batch~ : Int, heads~ : Int, head_dim~ : Int, time~ : Int)
  MeanHeadsPool(
    batch~ : Int,
    heads~ : Int,
    head_dim~ : Int,
    output_channels~ : Int,
    time~ : Int
  )
  GeluMultiply(elements~ : Int64)
  NormalizeRows(batch~ : Int, channels~ : Int, time~ : Int)
  Zeros(elements~ : Int64)
  VideoSlice(
    frames~ : Int,
    height~ : Int,
    width~ : Int,
    start_frame~ : Int,
    start_y~ : Int,
    start_x~ : Int,
    tile_frames~ : Int,
    tile_height~ : Int,
    tile_width~ : Int
  )
  Conv3d(
    input_channels~ : Int,
    output_channels~ : Int,
    frames~ : Int,
    height~ : Int,
    width~ : Int,
    kernel~ : Int,
    temporal_stride~ : Int,
    spatial_stride~ : Int,
    spatial_before~ : Int,
    spatial_after~ : Int
  )
  GroupNormSilu(
    channels~ : Int,
    frames~ : Int,
    height~ : Int,
    width~ : Int,
    groups~ : Int
  )
  InsertTile(
    channels~ : Int,
    frames~ : Int,
    height~ : Int,
    width~ : Int,
    tile_height~ : Int,
    tile_width~ : Int,
    y~ : Int,
    x~ : Int,
    copy_height~ : Int,
    copy_width~ : Int,
    overlap_y~ : Int,
    overlap_x~ : Int
  )
  InsertTime(
    channels~ : Int,
    frames~ : Int,
    height~ : Int,
    width~ : Int,
    tile_frames~ : Int,
    start~ : Int,
    count~ : Int
  )
  GaussianPatch(channels~ : Int, frames~ : Int, height~ : Int, width~ : Int)
} derive(Eq, Debug)

///|
pub(all) enum Origin {
  Input
  Weight(String)
  Temporary
} derive(Eq, Debug)

///|
pub(all) struct Value {
  elements : Int64
  origin : Origin
} derive(Eq, Debug)

///|
pub(all) struct Node {
  operation : Operation
  inputs : ReadOnlyArray[Int]
  output : Int
} derive(Eq, Debug)

///|
pub(all) suberror PlanError {
  InvalidGeometry
  InvalidGraph
  BudgetExceeded
} derive(Eq, Debug)

///|
fn product(values : ArrayView[Int]) -> Int64 raise PlanError {
  for value in values; acc = 1L {
    guard value > 0 && acc <= 2147483647L / value.to_int64() else {
      raise InvalidGeometry
    }
    continue acc * value.to_int64()
  } nobreak {
    acc
  }
}

///|
pub fn Operation::signature(
  self : Self,
) -> (ReadOnlyArray[Int64], Int64) raise PlanError {
  match self {
    Pad(batch~, input_time~, output_time~) => {
      guard output_time >= input_time else { raise InvalidGeometry }
      ([product([batch, input_time])], product([batch, output_time]))
    }
    WeightNorm(outputs~, per_output~) =>
      (
        [product([outputs]), product([outputs, per_output])],
        product([outputs, per_output]),
      )
    Conv(
      batch~,
      input_channels~,
      output_channels~,
      input_time~,
      output_time~,
      kernel~,
      dilation~,
      stride~,
      padding~
    ) => {
      guard dilation > 0 &&
        stride > 0 &&
        padding >= 0 &&
        input_time.to_int64() + 2L * padding.to_int64() >=
        dilation.to_int64() * (kernel.to_int64() - 1L) + 1L &&
        (
          input_time.to_int64() +
          2L * padding.to_int64() -
          dilation.to_int64() * (kernel.to_int64() - 1L) -
          1L
        ) /
        stride.to_int64() +
        1L ==
        output_time.to_int64() else {
        raise InvalidGeometry
      }
      (
        [
          product([batch, input_channels, input_time]),
          product([output_channels, input_channels, kernel]),
          product([output_channels]),
        ],
        product([batch, output_channels, output_time]),
      )
    }
    Snake(batch~, channels~, time~) =>
      (
        [product([batch, channels, time]), product([channels])],
        product([batch, channels, time]),
      )
    Add(elements~) | GeluMultiply(elements~) => {
      guard elements > 0L && elements <= 2147483647L else {
        raise InvalidGeometry
      }
      ([elements, elements], elements)
    }
    LayerNorm(batch~, channels~, time~) =>
      (
        [
          product([batch, channels, time]),
          product([channels]),
          product([channels]),
        ],
        product([batch, channels, time]),
      )
    Dense(batch~, input_channels~, output_channels~, time~) =>
      (
        [
          product([batch, input_channels, time]),
          product([output_channels, input_channels]),
          product([output_channels]),
        ],
        product([batch, output_channels, time]),
      )
    Qkv(batch~, channels~, time~) =>
      (
        [
          product([batch, channels, time]),
          product([3, channels, channels]),
          product([channels]),
          product([channels]),
          product([channels]),
        ],
        product([batch, 3, channels, time]),
      )
    CausalAttention(batch~, heads~, head_dim~, time~) =>
      (
        [product([batch, 3, heads, head_dim, time])],
        product([batch, heads, head_dim, time]),
      )
    MeanHeadsPool(batch~, heads~, head_dim~, output_channels~, time~) => {
      guard output_channels > 0 && head_dim % output_channels == 0 else {
        raise InvalidGeometry
      }
      (
        [product([batch, heads, head_dim, time])],
        product([batch, output_channels, time]),
      )
    }
    NormalizeRows(batch~, channels~, time~) =>
      (
        [
          product([batch, channels, time]),
          product([channels]),
          product([channels]),
        ],
        product([batch, channels, time]),
      )
    _ => video_signature(self)
  }
}

///|
pub struct Graph {
  values : ReadOnlyArray[Value]
  nodes : ReadOnlyArray[Node]
  output : Int
  offsets : ReadOnlyArray[Int64]
  scratch_bytes : Int64
}

///|
/// Validates SSA and assigns 256-byte aligned temporary intervals. An input
/// lives through its consuming launch, so producer and consumer cannot alias.
pub fn Graph::new(
  values : ArrayView[Value],
  nodes : ArrayView[Node],
  output~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise PlanError {
  guard values.length() > 0 &&
    nodes.length() > 0 &&
    output >= 0 &&
    output < values.length() &&
    max_scratch_bytes > 0L else {
    raise InvalidGraph
  }
  let defined = values.map(v => v.origin != Temporary)
  let last = Array::make(values.length(), -1)
  let birth = Array::make(values.length(), -1)
  for value in values {
    guard value.elements > 0L && value.elements <= 2147483647L else {
      raise InvalidGeometry
    }
  }
  for i, node in nodes {
    let (args, result) = node.operation.signature()
    guard node.inputs.length() == args.length() &&
      node.output >= 0 &&
      node.output < values.length() &&
      !defined[node.output] &&
      values[node.output].elements == result else {
      raise InvalidGraph
    }
    for j, input in node.inputs {
      guard input >= 0 &&
        input < values.length() &&
        defined[input] &&
        values[input].elements == args[j] else {
        raise InvalidGraph
      }
      last[input] = i
    }
    defined[node.output] = true
    birth[node.output] = i
  }
  guard defined[output] && values[output].origin == Temporary else {
    raise InvalidGraph
  }
  last[output] = nodes.length()
  let offsets = Array::make(values.length(), -1L)
  let sizes = values.map(v => (v.elements * 4L + 255L) / 256L * 256L)
  let mut high = 0L
  for node in nodes {
    let v = node.output
    let mut offset = 0L
    for ;; {
      let mut next = offset
      for j in 0..<values.length() {
        if offsets[j] >= 0L &&
          last[j] >= birth[v] &&
          birth[j] <= birth[v] &&
          offset < offsets[j] + sizes[j] &&
          offsets[j] < offset + sizes[v] {
          next = Int64::max(next, offsets[j] + sizes[j])
        }
      }
      if next == offset {
        break
      }
      offset = next
    }
    guard offset <= max_scratch_bytes && sizes[v] <= max_scratch_bytes - offset else {
      raise BudgetExceeded
    }
    offsets[v] = offset
    high = Int64::max(high, offset + sizes[v])
  }
  {
    values: ReadOnlyArray::from_array(values.to_owned()),
    nodes: ReadOnlyArray::from_array(nodes.to_owned()),
    output,
    offsets: ReadOnlyArray::from_array(offsets),
    scratch_bytes: high,
  }
}

///|
/// Immutable dense BF16 projection with F32 accumulation. Operands are
/// argument indices in an already leased ordered record. Rows are exact,
/// never a padded capacity. Backend implementation is private to device.
pub struct OrderedBf16Projection {
  priv launch_index : Int
  priv rows : Int
  priv columns : Int
  priv inner : Int
  priv input_argument : Int
  priv weight_argument : Int
  priv output_argument : Int
} derive(Debug, Eq)

///|
pub fn OrderedBf16Projection::new(
  launch_index~ : Int,
  rows~ : Int,
  columns~ : Int,
  inner~ : Int,
  input_argument~ : Int,
  weight_argument~ : Int,
  output_argument~ : Int,
) -> OrderedBf16Projection raise DeviceError {
  if launch_index < 0 ||
    rows <= 0 ||
    columns <= 0 ||
    inner <= 0 ||
    input_argument < 0 ||
    weight_argument < 0 ||
    output_argument < 0 ||
    input_argument == weight_argument ||
    input_argument == output_argument ||
    weight_argument == output_argument {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  {
    launch_index,
    rows,
    columns,
    inner,
    input_argument,
    weight_argument,
    output_argument,
  }
}

///|
fn ordered_projection_records(
  launches : ArrayView[OrderedKernelLaunch],
  projections : ArrayView[OrderedBf16Projection],
) -> FixedArray[Int] raise DeviceError {
  let records = FixedArray::make(launches.length() * 6, 0)
  for p in projections {
    if p.launch_index >= launches.length() || records[p.launch_index * 6] != 0 {
      raise InvalidArgument(OrderedExecutorCreate)
    }
    let arguments = launches[p.launch_index].arguments.length
    if p.input_argument >= arguments ||
      p.weight_argument >= arguments ||
      p.output_argument >= arguments {
      raise InvalidArgument(OrderedExecutorCreate)
    }
    let start = p.launch_index * 6
    records[start] = p.rows
    records[start + 1] = p.columns
    records[start + 2] = p.inner
    records[start + 3] = p.input_argument
    records[start + 4] = p.weight_argument
    records[start + 5] = p.output_argument
  }
  records
}

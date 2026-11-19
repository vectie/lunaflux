///|
/// Immutable dense BF16 projection with F32 accumulation. Operands are
/// argument indices in an already leased ordered record. Rows are exact,
/// never a padded capacity. Backend implementation is private to device.
pub struct OrderedBf16Projection {
  priv launch_index : Int32
  priv rows : Int32
  priv columns : Int32
  priv inner : Int32
  priv input_argument : Int32
  priv weight_argument : Int32
  priv output_argument : Int32
} derive(Debug, Eq)

///|
pub fn OrderedBf16Projection::new(
  launch_index~ : Int32,
  rows~ : Int32,
  columns~ : Int32,
  inner~ : Int32,
  input_argument~ : Int32,
  weight_argument~ : Int32,
  output_argument~ : Int32,
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
) -> FixedArray[Int32] raise DeviceError {
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

///|
pub extend OrderedBf16Projection with @debug.Debug::{to_repr}

///|
pub extend OrderedBf16Projection with Eq::{not_equal, equal}

///|
#borrow(context, stream, functions, dimensions, argument_starts, allocations, offsets, byte_counts, alignments, projections, status)
extern "c" fn raw_ordered_executor_create_with_projections(
  context : Context,
  stream : Stream,
  functions : FixedArray[Function],
  dimensions : FixedArray[Int32],
  argument_starts : FixedArray[Int32],
  allocations : FixedArray[Allocation],
  offsets : FixedArray[Int64],
  byte_counts : FixedArray[Int64],
  alignments : FixedArray[Int64],
  projections : FixedArray[Int32],
  policy : Int32,
  status : Ref[Int32],
) -> OrderedKernelExecutor = "lunaflux_cuda_ordered_executor_create_with_projections"

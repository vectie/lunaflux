///|
#borrow(context, stream, functions, dimensions, argument_starts, allocations, offsets, byte_counts, alignments, projections, status)
extern "c" fn raw_ordered_executor_create_with_projections(
  context : Context,
  stream : Stream,
  functions : FixedArray[Function],
  dimensions : FixedArray[Int],
  argument_starts : FixedArray[Int],
  allocations : FixedArray[Allocation],
  offsets : FixedArray[Int64],
  byte_counts : FixedArray[Int64],
  alignments : FixedArray[Int64],
  projections : FixedArray[Int],
  policy : Int,
  status : Ref[Int],
) -> OrderedKernelExecutor = "lunaflux_cuda_ordered_executor_create_with_projections"

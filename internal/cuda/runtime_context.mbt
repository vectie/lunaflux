///|
#borrow(status)
extern "c" fn raw_runtime_context_create(
  ordinal : Int,
  status : Ref[Int],
) -> Context = "lunaflux_cuda_runtime_context_create"

///|
/// Retain/release (never reset) the runtime's primary context. Required by
/// libraries whose asynchronous worker threads use cudaSetDevice internally.
pub fn open_runtime_context(ordinal : Int) -> Context raise CudaError {
  let status = Ref(0)
  let context = raw_runtime_context_create(ordinal, status)
  check_status(status.val, ContextCreate)
  context
}

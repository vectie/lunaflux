///|
/// Opens one process-visible CUDA device context. Every resource created from
/// it requires deterministic close in reverse dependency order.
pub fn open_context(ordinal : Int) -> Context raise DeviceError {
  let raw = @cuda.open_context(ordinal) catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn Context::close(self : Context) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Context::create_stream(self : Context) -> Stream raise DeviceError {
  let raw = self.raw.create_stream() catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn Stream::close(self : Stream) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Stream::synchronize(self : Stream) -> Unit raise DeviceError {
  self.raw.synchronize() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Context::create_event(self : Context) -> Event raise DeviceError {
  let raw = self.raw.create_event() catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn Event::close(self : Event) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Event::record(self : Event, stream : Stream) -> Unit raise DeviceError {
  self.raw.record(stream.raw) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Event::synchronize(self : Event) -> Unit raise DeviceError {
  self.raw.synchronize() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Event::elapsed_milliseconds(
  self : Event,
  finish : Event,
) -> Double raise DeviceError {
  self.raw.elapsed_milliseconds(finish.raw) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Context::allocate(
  self : Context,
  byte_count : Int64,
) -> Allocation raise DeviceError {
  let raw = self.raw.allocate(byte_count) catch {
    error => raise error_from_cuda(error)
  }
  { raw, byte_count }
}

///|
pub fn Allocation::close(self : Allocation) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Synchronously copies a bounded host range into this allocation. Synchronous
/// transfer keeps the borrowed MoonBit buffer from escaping the FFI call.
pub fn Allocation::copy_from_host(
  self : Allocation,
  source : Bytes,
  source_offset : Int64,
  destination_offset : Int64,
  byte_count : Int64,
) -> Unit raise DeviceError {
  self.raw.copy_from_host_at(
    source, source_offset, destination_offset, byte_count,
  ) catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Synchronously copies a bounded device range into a fresh host buffer.
pub fn Allocation::copy_to_host(
  self : Allocation,
  source_offset : Int64,
  byte_count : Int64,
) -> Bytes raise DeviceError {
  self.raw.copy_to_host_at(source_offset, byte_count) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Context::load_module(
  self : Context,
  image : Bytes,
) -> Module raise DeviceError {
  let raw = self.raw.load_module(image) catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn Module::close(self : Module) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Module::load_function(
  self : Module,
  name : Bytes,
) -> Function raise DeviceError {
  let raw = self.raw.load_function(name) catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn Function::close(self : Function) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Context::create_cublas_lt(self : Context) -> CublasLt raise DeviceError {
  let raw = self.raw.create_cublas_lt() catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn CublasLt::close(self : CublasLt) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Creates a reusable descriptor for row-major
/// `output[rows, columns] = left[rows, inner] * right[inner, columns]`.
/// Inputs and output use BF16 storage; accumulation uses FP32.
pub fn CublasLt::create_bf16_gemm_plan(
  self : CublasLt,
  rows : Int64,
  inner : Int64,
  columns : Int64,
  workspace_byte_count : Int64,
) -> Bf16GemmPlan raise DeviceError {
  let raw = self.raw.create_bf16_gemm_plan(
    rows, inner, columns, workspace_byte_count,
  ) catch {
    error => raise error_from_cuda(error)
  }
  { raw, rows, inner, columns, workspace_byte_count }
}

///|
pub fn Bf16GemmPlan::close(self : Bf16GemmPlan) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Runs this plan on `stream` and synchronizes it before returning. Phase 1
/// deliberately uses synchronous completion so every borrowed operand remains
/// live for the complete device operation. Offsets are checked against their
/// allocation ranges before entering the native ABI.
pub fn Bf16GemmPlan::run(
  self : Bf16GemmPlan,
  left : Allocation,
  left_offset : Int64,
  right : Allocation,
  right_offset : Int64,
  output : Allocation,
  output_offset : Int64,
  workspace : Allocation,
  workspace_offset : Int64,
  stream : Stream,
) -> Unit raise DeviceError {
  self.raw.run(
    left.raw,
    left_offset,
    right.raw,
    right_offset,
    output.raw,
    output_offset,
    workspace.raw,
    workspace_offset,
    stream.raw,
  ) catch {
    error => raise error_from_cuda(error)
  }
}

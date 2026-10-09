///|
/// Four admitted functions in staged_joint_attention_bf16_source ABI order.
pub struct AttentionFunctions {
  priv qkv : @device.Function
  priv norm : @device.Function
  priv attention : @device.Function
  priv output : @device.Function
}

///|
pub fn AttentionFunctions::new(
  qkv : @device.Function,
  norm : @device.Function,
  attention : @device.Function,
  output : @device.Function,
) -> Self {
  { qkv, norm, attention, output, }
}

///|
/// Bind one complete staged attention block, without submitting work. Scratch
/// is four consecutive BF16 [rows, heads*head_dimension] tensors: Q,K,V,A.
/// The staged normalization renderer reads an entire head into private arrays
/// before writing, so normalized Q/K safely replace their own input tensors.
/// Weights are Q,K,V matrices, Q/K norm vectors, then the output matrix, all
/// BF16. Caller owns allocations and must serialize users of the same scratch.
/// The attention function must encode the prepared document boundaries.
pub fn RotaryBindings::attention_launches(
  self : Self,
  functions : AttentionFunctions,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  scratch~ : @device.Allocation,
  scratch_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  hidden_width~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  functions.launches(
    rows=self.rows,
    rotary=self,
    input~,
    input_offset~,
    output~,
    output_offset~,
    scratch~,
    scratch_offset~,
    weights~,
    weight_offset~,
    hidden_width~,
    head_count~,
    head_dimension~,
  )
}

///|
/// Omit rotary only for a function compiled with apply_rotary=false (six QK
/// norm operands). Rotated functions retain the eight-operand ABI.
pub fn AttentionFunctions::launches(
  self : Self,
  rows~ : Int,
  rotary? : RotaryBindings,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  scratch~ : @device.Allocation,
  scratch_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  hidden_width~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let functions = self
  let scratch_bytes = staged_attention_scratch_bytes(
    rows~,
    head_count~,
    head_dimension~,
  )
  let tensor_bytes = scratch_bytes / 4L
  let norm_bytes = head_dimension.to_int64() * 2L
  if rotary is Some(binding) {
    guard binding.rows == rows else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    ignore(qk_rotary_geometry(rows, binding.width, head_count, head_dimension))
  }
  let norm_dimensions = linear_dimensions(
    rows.to_int64() * head_count.to_int64(),
    1,
  )
  let (hidden_bytes, matrix_bytes) = attention_projection_geometry(
    rows, hidden_width, head_count, head_dimension,
  )
  fn region(
    allocation : @device.Allocation,
    offset : Int64,
    bytes : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      allocation,
      offset~,
      byte_count=bytes,
      alignment=2L,
    )
  }
  // Check whole spans before offset arithmetic and before constructing any
  // launch. No queue or native resource is acquired by this preparation step.
  ignore(
    region(
      scratch,
      scratch_offset,
      staged_attention_scratch_bytes(rows~, head_count~, head_dimension~),
    ),
  )
  ignore(region(weights, weight_offset, 4L * matrix_bytes + 2L * norm_bytes))
  let q = region(scratch, scratch_offset, tensor_bytes)
  let k = region(scratch, scratch_offset + tensor_bytes, tensor_bytes)
  let v = region(scratch, scratch_offset + 2L * tensor_bytes, tensor_bytes)
  let attended = region(
    scratch,
    scratch_offset + 3L * tensor_bytes,
    tensor_bytes,
  )
  let qkv = @device.OrderedKernelLaunch::new(
    functions.qkv,
    linear_dimensions(tensor_bytes / 2L, 256),
    @device.KernelArguments::new([
      region(input, input_offset, hidden_bytes),
      region(weights, weight_offset, matrix_bytes),
      region(weights, weight_offset + matrix_bytes, matrix_bytes),
      region(weights, weight_offset + 2L * matrix_bytes, matrix_bytes),
      q,
      k,
      v,
    ]),
  )
  let rotary_arguments = match rotary {
    Some(binding) => [binding.cosine(), binding.sine()]
    None => []
  }
  let norm_arguments = normalization_arguments(
    q,
    k,
    region(weights, weight_offset + 3L * matrix_bytes, norm_bytes),
    region(weights, weight_offset + 3L * matrix_bytes + norm_bytes, norm_bytes),
    rotary_arguments,
  )
  let norm = @device.OrderedKernelLaunch::new(
    functions.norm,
    norm_dimensions,
    @device.KernelArguments::new(norm_arguments),
  )
  let attention = @device.OrderedKernelLaunch::new(
    functions.attention,
    linear_dimensions(tensor_bytes / 2L, 256),
    @device.KernelArguments::new([q, k, v, attended]),
  )
  let projection = @device.OrderedKernelLaunch::new(
    functions.output,
    linear_dimensions(hidden_bytes / 2L, 256),
    @device.KernelArguments::new([
      attended,
      region(
        weights,
        weight_offset + 3L * matrix_bytes + 2L * norm_bytes,
        matrix_bytes,
      ),
      region(output, output_offset, hidden_bytes),
    ]),
  )
  [qkv, norm, attention, projection]
}

///|
fn[T] normalization_arguments(
  q : T,
  k : T,
  qw : T,
  kw : T,
  rotary : ArrayView[T],
) -> Array[T] raise @device.DeviceError {
  guard rotary.length() == 0 || rotary.length() == 2 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let args = [q, k, qw, kw]
  args.append(rotary)
  args.push(q)
  args.push(k)
  args
}

///|
fn attention_projection_geometry(
  rows : Int,
  hidden : Int,
  heads : Int,
  dimension : Int,
) -> (Int64, Int64) raise @device.DeviceError {
  guard rows > 0 && hidden > 0 && heads > 0 && dimension > 0 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let inner = heads.to_int64() * dimension.to_int64()
  guard inner <= 2147483647L &&
    rows.to_int64() <= 2147483647L / hidden.to_int64() &&
    inner <= 2147483647L / hidden.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  (rows.to_int64() * hidden.to_int64() * 2L, inner * hidden.to_int64() * 2L)
}

///|
/// Required reusable scratch for the current staged BF16 attention ABI.
/// This is an allocation size, not an estimate of compiler-local spills.
pub fn staged_attention_scratch_bytes(
  rows~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
) -> Int64 raise @device.DeviceError {
  guard rows > 0 && head_count > 0 && head_dimension > 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let row_heads = rows.to_int64() * head_count.to_int64()
  guard row_heads <= 2147483647L / head_dimension.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  row_heads * head_dimension.to_int64() * 8L
}

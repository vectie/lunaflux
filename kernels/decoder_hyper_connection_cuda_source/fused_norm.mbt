///|
/// Fusion preserves the observable BF16 collapse and the declared normalization
/// tree. Only launch/publication placement changes, not floating-point order.
fn fused_collapse_norm_body(
  precision : @precision.HyperConnectionPrecision,
  workers : Int,
) -> String {
  let (_, hidden, _) = precision.geometry()
  let expression = (
    operation : @precision.RmsExpression,
    bindings : ArrayView[String],
  ) => {
    @arithmetic.scalar_expression(operation.program(), bindings) catch {
      _ => abort("invalid closed RMS expression")
    }
  }
  let square = expression(SquareUpdate, ["square_sum", "value"])
  let add = expression(SumUpdate, ["partial[worker]", "partial[worker + span]"])
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    precision.normalization_epsilon(),
  )
  let variance = expression(Variance, ["partial[0]", "\{hidden}.0f", eps])
  let inverse = @arithmetic.scalar_expression(
    @precision.ReciprocalRootLaw::ReciprocalSqrtFloat32.expression(),
    [variance],
  ) catch {
    _ => abort("invalid RMS root expression")
  }
  let publish = expression(WeightedPublish, [
    "__bfloat162float(collapsed[base_index + column])", "norm_inverse", "__bfloat162float(norm_weight[column])",
  ])
  (
    $|  __syncthreads(); // publish rounded BF16 collapse before any norm reader
    $|  __shared__ float partial[\{workers}];
    $|  const unsigned long long base_index = (unsigned long long)row * \{hidden};
    $|  float square_sum = 0.0f;
    $|  for (int column = worker; column < \{hidden}; column += \{workers}) {
    $|    const float value = __bfloat162float(collapsed[base_index + column]);
    $|    square_sum = \{square};
    $|  }
    $|  partial[worker] = square_sum; __syncthreads();
    $|  for (int span = \{workers / 2}; span > 0; span >>= 1) {
    $|    if (worker < span) partial[worker] = \{add};
    $|    __syncthreads();
    $|  }
    $|  const float norm_inverse = \{inverse};
    $|  for (int column = worker; column < \{hidden}; column += \{workers}) normalized[base_index + column] = \{publish};
  )
}

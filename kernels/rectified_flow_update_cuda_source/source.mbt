///|
pub(all) suberror SourceError {
  InvalidGeometry
}

///|
/// F32 eta-zero RF update in a fixed symbol. Coefficients contain exactly
/// [1-f32(timestep), f32(next_sigma)/f32(sigma), 1-ratio]. They are prepared
/// once per step, outside the element loop. Exact in-place state/output alias
/// is allowed; partial overlap and coefficient/output overlap are not.
pub fn render(elements~ : Int64, block_size~ : Int) -> Bytes raise SourceError {
  guard elements > 0L &&
    elements <= 2147483647L &&
    block_size > 0 &&
    block_size <= 1024 else {
    raise InvalidGeometry
  }
  let source =
    $|extern "C" __global__ void lunaflux_rectified_flow_f32_eta0_v1(
    $|    const float *state, const float *velocity,
    $|    const float *coefficients, float *output) {
    $|  const unsigned long long i = (unsigned long long)blockIdx.x * \{block_size}ULL + threadIdx.x;
    $|  if (i >= \{elements}ULL) return;
    $|  const float x = state[i];
    $|  const float scaled = __fmul_rn(coefficients[0], velocity[i]);
    $|  const float denoised = __fadd_rn(x, scaled);
    $|  const float retained = __fmul_rn(coefficients[1], x);
    $|  const float predicted = __fmul_rn(coefficients[2], denoised);
    $|  output[i] = __fadd_rn(retained, predicted);
    $|}
    $|
  @utf8.encode(source)
}

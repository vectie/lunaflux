///|
pub(all) suberror SourceError {
  InvalidBlockSize
}

///|
/// Bit-preserving BF16 row assembly with explicit zero padding. Inputs are
/// compact source matrices; map is Plan.mapping(), uploaded once and immutable.
/// Output must not overlap any nonempty input. No arithmetic on BF16 values.
pub fn render(
  plan : @packed_rows_plan.Plan,
  block_size~ : Int,
) -> Bytes raise SourceError {
  guard block_size > 0 && block_size <= 1024 else { raise InvalidBlockSize }
  @utf8.encode(
    (
      $|extern "C" __global__ void lunaflux_packed_rows_bf16_v1(
      $|  const unsigned short *conditioning, const unsigned short *video,
      $|  const unsigned short *audio, const unsigned int *map, unsigned short *output) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * \{block_size}ULL + threadIdx.x;
      $|  if (i >= \{plan.element_count()}ULL) return;
      $|  const unsigned long long row = i / \{plan.width()}ULL;
      $|  const unsigned int tag = map[row * 2ULL];
      $|  if (tag == 0U) { output[i] = 0; return; }
      $|  const unsigned long long source = (unsigned long long)map[row * 2ULL + 1ULL] * \{plan.width()}ULL + i % \{plan.width()}ULL;
      $|  const unsigned short *input = tag == 1U ? conditioning : (tag == 2U ? video : audio);
      $|  output[i] = input[source];
      $|}
      $|
    ),
  )
}

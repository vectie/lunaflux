///|
pub(all) suberror SourceError {
  InvalidBlockSize
}

///|
/// Out-of-place bit-preserving permutation of 32-bit cells. Input/output
/// must each contain plan.element_count() cells in non-overlapping storage.
/// No scratch, synchronization, host index upload, or floating-point conversion.
/// Launch ceil(elements/block_size) blocks with exactly block_size threads.
pub fn render(
  plan : @dense_permutation_plan.Permutation,
  block_size~ : Int,
) -> Bytes raise SourceError {
  guard block_size > 0 && block_size <= 1024 else { raise InvalidBlockSize }
  let source = StringBuilder()
  source <+
    $|extern "C" __global__ void lunaflux_dense_permutation_u32_v1(
    $|    const unsigned int *input, unsigned int *output) {
    $|  const unsigned long long i = (unsigned long long)blockIdx.x * \{block_size}ULL + threadIdx.x;
    $|  if (i >= \{plan.element_count()}ULL) return;
    $|
  if plan.is_identity() {
    source.write_string("  output[i] = input[i];\n")
  } else {
    source.write_string("  unsigned long long src = 0ULL;\n")
    for term in plan.terms() {
      // All divisors are compile-time constants; no per-thread shape loads.
      let coordinate = if term.output_stride == 1L {
        "(i % \{term.extent}ULL)"
      } else {
        "((i / \{term.output_stride}ULL) % \{term.extent}ULL)"
      }
      if term.input_stride == 1L {
        source <+ "  src += \{coordinate};\n"
      } else {
        source <+ "  src += \{coordinate} * \{term.input_stride}ULL;\n"
      }
    }
    source.write_string("  output[i] = input[src];\n")
  }
  source.write_string("}\n")
  @utf8.encode(source.to_string())
}

///|
/// WMMA instruction realization of the same physical product used by packed
/// projections. Column ownership is a backend binding; SSA sharing and
/// lookahead action order are supplied by the physical program.
fn projection_head_fragment_source(
  program : @physical_tile_ir.FragmentProgram,
  input_elements : Int,
) -> String {
  let (rows, live, width, quantum, fragments) = program.geometry()
  guard rows > 0 && quantum == 16 else {
    abort("unsupported head fragment binding")
  }
  let load = fn(offset : String, slot : String) -> String {
    (
      $|        if ((window * LF_LIVE_COLUMNS * LF_WARPS + warp) * 16 < LF_HEAD_DIM) {
      $|          #pragma unroll
      $|          for (int row_fragment=0; row_fragment<\{rows}; ++row_fragment)
      $|            nvcuda::wmma::load_matrix_sync(a[\{slot}][row_fragment], current + row_fragment * 16 * \{width} + \{offset}, \{width});
      $|        }
      $|        #pragma unroll
      $|        for (int column_slot = 0; column_slot < \{live}; ++column_slot) {
      $|          const int column_base = ((window * LF_LIVE_COLUMNS + column_slot) * LF_WARPS + warp) * 16;
      $|          if (column_base < LF_HEAD_DIM) nvcuda::wmma::load_matrix_sync(b[\{slot}][column_slot], current + \{input_elements} + column_base * \{width} + \{offset}, \{width});
      $|        }
    )
  }
  let iteration = program
    .iteration()
    .map(action => {
      match action {
        ReadCurrent => load("fragment_inner", "0")
        ReadNextIfPresent =>
          "if (fragment_inner + \{quantum} < \{width}) {\n" +
          load("fragment_inner + \{quantum}", "fragment_stage ^ 1") +
          "\n}"
        AccumulateCurrent =>
          (
            $|        #pragma unroll
            $|        for (int column_slot = 0; column_slot < \{live}; ++column_slot) {
            $|          const int column_base = ((window * LF_LIVE_COLUMNS + column_slot) * LF_WARPS + warp) * 16;
            $|          #pragma unroll
            $|          for (int row_fragment=0; row_fragment<\{rows}; ++row_fragment)
            $|            if (column_base < LF_HEAD_DIM) nvcuda::wmma::mma_sync(c[row_fragment][column_slot], a[fragment_stage][row_fragment], b[fragment_stage][column_slot], c[row_fragment][column_slot]);
            $|        }
          )
      }
    })
    .join("\n")
  (
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a[\{fragments}][\{rows}];
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b[\{fragments}][\{live}];
    $|      \{if program.primes_operands() { load("0", "0") } else { "" }}
    $|      int fragment_stage = 0;
    $|      #pragma unroll
    $|      for (int fragment_inner = 0; fragment_inner < \{width}; fragment_inner += \{quantum}) {
    $|        \{iteration}
    $|        \{if program.primes_operands() { "fragment_stage ^= 1;" } else { "" }}
    $|      }
  )
}

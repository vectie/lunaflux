///|
/// Packed instruction realization of the same physical product used by ordinary
/// projections. Column ownership is a backend binding; SSA sharing and
/// lookahead action order are supplied by the physical program.
fn projection_head_fragment_source(
  program : @physical_tile_ir.FragmentProgram,
  input_elements : Int,
  domain : @physical_tile_ir.FragmentRowDomain,
) -> String {
  let (rows, live, width, quantum, fragments) = program.geometry()
  let (domain_rows, row_quantum) = domain.geometry()
  guard rows > 0 && quantum == 16 && domain_rows == rows && row_quantum == 16 else {
    abort("unsupported head fragment binding")
  }
  let load = fn(offset : String, slot : String) -> String {
    (
      $|        if ((window * LF_LIVE_COLUMNS * LF_WARPS + warp) * 16 < LF_PROJECTION_WIDTH) {
      $|          #pragma unroll
      $|          for (int row_fragment=0; row_fragment<\{rows}; ++row_fragment)
      $|          {
      $|            if (row_fragment >= lf_active_row_fragments) break;
      $|            unsigned *operand=a[\{slot}][row_fragment];
      $|            const unsigned address=(unsigned)__cvta_generic_to_shared(current + (row_fragment*16+lane%16)*\{width} + \{offset} + lane/16*8);
      $|            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(operand[0]),"=r"(operand[1]),"=r"(operand[2]),"=r"(operand[3]) : "r"(address));
      $|          }
      $|        }
      $|        #pragma unroll
      $|        for (int column_slot = 0; column_slot < \{live}; ++column_slot) {
      $|          const int column_base = ((window * LF_LIVE_COLUMNS + column_slot) * LF_WARPS + warp) * 16;
      $|          if (column_base < LF_PROJECTION_WIDTH) {
      $|            unsigned *operand=b[\{slot}][column_slot];
      $|            const unsigned address=(unsigned)__cvta_generic_to_shared(current + \{input_elements} + (column_base+lane%16)*\{width} + \{offset} + lane/16*8);
      $|            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(operand[0]),"=r"(operand[1]),"=r"(operand[2]),"=r"(operand[3]) : "r"(address));
      $|          } else {
      $|            // Joint products include inactive column owners. Bind them
      $|            // to a defined zero operand; their outputs are never stored.
      $|            #pragma unroll
      $|            for (int word=0; word<4; ++word) b[\{slot}][column_slot][word]=0U;
      $|          }
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
        AccumulateCurrent => {
          // Joint register binding is retained for each live prefix. Do not
          // turn this into one asm region per MMA and reintroduce tuple MOVs.
          let branches = Array::makei(rows, index => {
            let active = rows - index
            "\{if index == 0 { "if" } else { "else if" }} (lf_active_row_fragments == \{active}) {\n" +
            projection_head_joint_fold_source(active, live) +
            "\n}"
          })
          branches.join("\n")
        }
      }
    })
    .join("\n")
  (
    $|      constexpr int lf_active_row_fragments = decltype(live_row_domain)::value;
    $|      unsigned a[\{fragments}][\{rows}][4];
    $|      unsigned b[\{fragments}][\{live}][4];
    $|      \{if program.primes_operands() { load("0", "0") } else { "" }}
    $|      int fragment_stage = 0;
    $|      #pragma unroll
    $|      for (int fragment_inner = 0; fragment_inner < \{width}; fragment_inner += \{quantum}) {
    $|        \{iteration}
    $|        \{if program.primes_operands() { "fragment_stage ^= 1;" } else { "" }}
    $|      }
  )
}

///|
/// Bind the complete immutable product fan-out in one instruction region.
/// Separate per-MMA constraints force the native compiler to repeatedly copy
/// carried accumulators into aligned register tuples. Joint binding lets the
/// allocator see all owners while retaining each accumulator's ordered fold.
fn projection_head_joint_fold_source(rows : Int, columns : Int) -> String {
  let outputs : Array[String] = []
  let inputs : Array[String] = []
  let instructions : Array[String] = []
  let input_base = rows * columns * 8
  for row in 0..<rows {
    for word in 0..<4 {
      inputs.push("\"r\"(a[fragment_stage][\{row}][\{word}])")
    }
  }
  for column in 0..<columns {
    for word in 0..<4 {
      inputs.push("\"r\"(b[fragment_stage][\{column}][\{word}])")
    }
  }
  for row in 0..<rows {
    for column in 0..<columns {
      for item in 0..<8 {
        outputs.push("\"+f\"(c[\{row}][\{column}][\{item}])")
      }
      for half in 0..<2 {
        let d = (row * columns + column) * 8 + half * 4
        let a = input_base + row * 4
        let b = input_base + rows * 4 + column * 4 + half
        instructions.push(
          "\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%\{d},%\{d+1},%\{d+2},%\{d+3}}, {%\{a},%\{a+1},%\{a+2},%\{a+3}}, {%\{b},%\{b+2}}, {%\{d},%\{d+1},%\{d+2},%\{d+3}};\"",
        )
      }
    }
  }
  (
    $|        if ((window * LF_LIVE_COLUMNS * LF_WARPS + warp) * 16 < LF_PROJECTION_WIDTH) {
    $|          // joint-head-products-v1: no numerical reassociation.
    $|          asm volatile(
    $|            \{instructions.join("\n            ")}
    $|            : \{outputs.join(",")}
    $|            : \{inputs.join(",")});
    $|        }
  )
}

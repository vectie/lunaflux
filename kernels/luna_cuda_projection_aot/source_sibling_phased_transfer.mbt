///|
/// Stage a product fold's shared operands through a bounded register batch.
/// Read and publish phases are thread-local: vectors have unique owners and
/// staging writes only the inactive shared slot. The caller's loop-end fence
/// publishes that slot before its next use. Synchronous loads/stores need no
/// async group operations or internal CTA fence. Compute shared addresses at
/// publication, so they do not lengthen the live register tuple. Tail rows
/// contribute zero without issuing an invalid global read.
fn projection_sibling_phased_transfer_source(
  rows : Int,
  columns : Int,
  width : Int,
  input_width : Int,
  threads : Int,
) -> String {
  let elements = (rows + columns * 2) * width
  let vectors = elements / 8
  let batches = (vectors + threads - 1) / threads
  let segments = @luna_projection_tile_compiler.partition_projection_transfer_segments(
    [rows * width / 8, columns * width / 8, columns * width / 8],
    workers=threads,
  ).unwrap()
  let addresses = StringBuilder()
  for segment in segments {
    let (base, row_base) = match segment.operand {
      0 => ("input", "row_begin")
      1 => ("gate_weight", "column_begin")
      _ => ("up_weight", "column_begin")
    }
    let guard_workers = segment.worker_begin != 0 ||
      segment.worker_end != threads
    addresses <+ "        {\n"
    if guard_workers {
      let predicate = if segment.worker_begin == 0 {
        "threadIdx.x < \{segment.worker_end}"
      } else if segment.worker_end == threads {
        "threadIdx.x >= \{segment.worker_begin}"
      } else {
        "threadIdx.x >= \{segment.worker_begin} && threadIdx.x < \{segment.worker_end}"
      }
      addresses <+ "          if (\{predicate}) {\n"
    }
    addresses <+
      "          const int element = ((int)threadIdx.x + (\{segment.operand_vector_origin})) * 8;\n"
    addresses <+
      "          const int source_row = \{row_base} + element / \{width};\n"
    if segment.operand == 0 {
      addresses <+ "          if (source_row < token_count) {\n"
    }
    let slot = segment.round
    addresses <+ "          transfer_active[\{slot}] = true;\n"
    addresses <+
      "          transfer_sources[\{slot}] = \{base} + (unsigned long long)source_row * \{input_width}ULL + element % \{width};\n"
    if segment.operand == 0 {
      addresses <+ "          }\n"
    }
    if guard_workers {
      addresses <+ "          }\n"
    }
    addresses <+ "        }\n"
  }
  (
    $|const __nv_bfloat16 *transfer_sources[\{batches}] = {};
    $|      bool transfer_active[\{batches}] = {};
    $|\{addresses.to_string()}
    $|      auto stage_operands = [&](int k_base, int stage) {
    $|        __nv_bfloat16 *destination = sibling_storage + stage * \{elements};
    $|        unsigned values[\{batches}][4] = {};
    $|        #pragma unroll
    $|        for (int slot=0;slot<\{batches};++slot) {
    $|          if (transfer_active[slot]) {
    $|            const __nv_bfloat16 *source=transfer_sources[slot]+k_base;
    $|            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(values[slot][0]),"=r"(values[slot][1]),"=r"(values[slot][2]),"=r"(values[slot][3]) : "l"(source) : "memory");
    $|          }
    $|        }
    $|        #pragma unroll
    $|        for (int slot = 0; slot < \{batches}; ++slot) {
    $|          if (threadIdx.x + slot * \{threads} < \{vectors}) {
    $|            const int element = (threadIdx.x + slot * \{threads}) * 8;
    $|            const unsigned address = (unsigned)__cvta_generic_to_shared(destination + operand_index(element / \{width}, element % \{width}));
    $|            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(address), "r"(values[slot][0]), "r"(values[slot][1]), "r"(values[slot][2]), "r"(values[slot][3]) : "memory");
    $|          }
    $|        }
    $|      };
  )
}

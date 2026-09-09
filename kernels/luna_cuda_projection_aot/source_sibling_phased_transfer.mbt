///|
/// Stage a product fold's shared operands through a bounded register batch.
/// All lanes finish the read phase before publishing shared vectors, and all
/// publishers finish before another phase may read global operands. The
/// immutable geometry determines both the batch capacity and lane ownership;
/// inactive tail rows contribute zero without issuing an invalid global read.
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
  (
    $|auto stage_operands = [&](int k_base, int stage) {
    $|        __nv_bfloat16 *destination = sibling_storage + stage * \{elements};
    $|        unsigned values[\{batches}][4] = {}, addresses[\{batches}] = {};
    $|        #pragma unroll
    $|        for (int slot = 0; slot < \{batches}; ++slot) {
    $|          const int vector = threadIdx.x + slot * \{threads};
    $|          if (vector < \{vectors}) {
    $|            const int element = vector * 8;
    $|            const bool is_input = element < \{rows * width};
    $|            const bool is_gate = element < \{(rows + columns) * width};
    $|            const int offset = is_input ? element : (is_gate ? element - \{rows * width} : element - \{(rows + columns) * width});
    $|            const int row = offset / \{width}, k = offset % \{width};
    $|            const bool valid = !is_input || row_begin + row < token_count;
    $|            const __nv_bfloat16 *base = is_input ? input : (is_gate ? gate_weight : up_weight);
    $|            const int source_row = is_input ? row_begin + row : column_begin + row;
    $|            addresses[slot] = (unsigned)__cvta_generic_to_shared(destination + operand_index(element / \{width}, k));
    $|            if (valid) {
    $|              const __nv_bfloat16 *source = base + (unsigned long long)source_row * \{input_width}ULL + k_base + k;
    $|              asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(values[slot][0]), "=r"(values[slot][1]), "=r"(values[slot][2]), "=r"(values[slot][3]) : "l"(source) : "memory");
    $|            }
    $|          }
    $|        }
    $|        __syncthreads();
    $|        #pragma unroll
    $|        for (int slot = 0; slot < \{batches}; ++slot) {
    $|          if (threadIdx.x + slot * \{threads} < \{vectors}) {
    $|            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(addresses[slot]), "r"(values[slot][0]), "r"(values[slot][1]), "r"(values[slot][2]), "r"(values[slot][3]) : "memory");
    $|          }
    $|        }
    $|        __syncthreads();
    $|        asm volatile("cp.async.commit_group;" ::: "memory");
    $|      };
  )
}

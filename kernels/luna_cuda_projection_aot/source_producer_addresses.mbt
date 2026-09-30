///|
/// Bind invariant producer coordinates once per independent output workgroup.
/// The fold varies only K and ring slot. Copies, waits and publication remain
/// at their original effect locations; no global operand value is hoisted.
fn projection_producer_address_source(
  plan : @luna_projection_tile_compiler.ProjectionProducerAddressPlan,
  name : String,
  global_base : String,
  storage : String,
  stride : Int,
  row_binding : String,
  validity : String,
  weight_binding : String,
) -> String {
  let (vectors, workers, rounds) = plan.geometry()
  (
    $|      const __nv_bfloat16 *\{name}_base[\{rounds}];
    $|      unsigned \{name}_shared[\{rounds}];
    $|      bool \{name}_valid[\{rounds}];
    $|      #pragma unroll
    $|      for (int address_slot = 0; address_slot < \{rounds}; ++address_slot) {
    $|        const int vector = (int)threadIdx.x + address_slot * \{workers};
    $|        if (vector < \{vectors}) {
    $|          \{row_binding}
    $|          \{weight_binding}
    $|          const bool valid = \{validity};
    $|          \{name}_valid[address_slot] = valid;
    $|          \{name}_base[address_slot] = valid ? \{global_base} + (unsigned long long)source_row * \{stride}ULL + k : \{global_base};
    $|          \{name}_shared[address_slot] = (unsigned)__cvta_generic_to_shared(\{storage} + operand_index(local_row, k));
    $|        }
    $|      }
  )
}

///|
fn projection_hoisted_transfer_source(
  plan : @luna_projection_tile_compiler.ProjectionProducerAddressPlan,
  name : String,
  slot_elements : Int,
  offset : String,
  asynchronous : Bool,
  initial_synchronous? : Bool = false,
) -> String {
  let (vectors, workers, rounds) = plan.geometry()
  let copy = if initial_synchronous {
    "if (overlap) { " +
    projection_operand_transfer_source(asynchronous~) +
    " } else { " +
    projection_operand_transfer_source(asynchronous=false) +
    " }"
  } else {
    projection_operand_transfer_source(asynchronous~)
  }
  (
    $|        #pragma unroll
    $|        for (int address_slot = 0; address_slot < \{rounds}; ++address_slot) {
    $|          if ((int)threadIdx.x + address_slot * \{workers} < \{vectors}) {
    $|            const bool valid = \{name}_valid[address_slot];
    $|            const __nv_bfloat16 *source = \{name}_base[address_slot] + \{offset};
    $|            const unsigned address = \{name}_shared[address_slot] + stage * \{slot_elements * 2}U;
    $|            \{copy}
    $|          }
    $|        }
  )
}

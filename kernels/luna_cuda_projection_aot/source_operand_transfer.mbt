///|
/// Instruction realization of the retained transport effect. Emission cannot
/// independently select asynchronous issue while the plan promises completion.
fn projection_operand_transfer_source(asynchronous~ : Bool) -> String {
  let transfer = if !asynchronous {
    (
      #|unsigned x, y, z, w;
      #|            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(source) : "memory");
      #|            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(address), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
    )
  } else {
    (
      #|asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(source) : "memory");
    )
  }
  (
    $|if (valid) {
    $|            \{transfer}
    $|          } else {
    $|            asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(address) : "memory");
    $|          }
  )
}

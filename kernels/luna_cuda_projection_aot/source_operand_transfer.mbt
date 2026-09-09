///|
/// Lower one aligned, disjoint vector write. A short source row splits a
/// bypass-copy warp into multiple global transactions even when its shared
/// destination banks are disjoint. Materialize that vector in registers for
/// narrow rows; full transport rows retain asynchronous copies. Invalid rows
/// are pure zero values, not zero-byte global-copy requests. Both branches
/// preserve the same shared layout and the surrounding stage ownership fence.
fn projection_operand_transfer_source(
  width : Int,
  asynchronous? : Bool = true,
) -> String {
  let transfer = if width < 64 || !asynchronous {
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

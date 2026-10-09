///|
pub fn visual_scatter_source(
  rows~ : Int,
  width~ : Int,
  prefix~ : String,
) -> Bytes raise TextEncoderSourceError {
  guard rows > 0 && rows <= 262144 && width > 0 && width <= 65536 else {
    raise InvalidShape
  }
  guard symbol_ok(prefix) else { raise InvalidSymbol }
  let out = StringBuilder()
  out.write_string("#include <cuda_bf16.h>\n#include <stdint.h>\n")
  for (suffix, add) in [("_visual_set", false), ("_visual_add", true)] {
    out.write_string(
      (
        $|extern "C" __global__ void \{prefix}\{suffix}(const int32_t *counts,const int32_t *positions,const __nv_bfloat16 *input,__nv_bfloat16 *state) {
        $| const unsigned long long i=(unsigned long long)blockIdx.x*256+threadIdx.x;
        $| if(counts[0]<=0 || counts[0]>\{rows} || i>=(unsigned long long)counts[0]*\{width}) return;
        $| const unsigned long long target=(unsigned long long)positions[i/\{width}]*\{width}+i%\{width};
      ),
    )
    out.write_string(
      if add {
        "state[target]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(state[target]),__bfloat162float(input[i])));\n}\n"
      } else {
        "state[target]=input[i];\n}\n"
      },
    )
  }
  @utf8.encode(out.to_string())
}

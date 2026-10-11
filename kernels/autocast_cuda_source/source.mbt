///|
/// Refine backend-private 16-bit storage primitives in an AOT template.
/// F32 accumulation and operation boundaries are retained verbatim. This
/// renderer has no model names, shape policy or runtime side effects.
pub fn storage_source(
  source : String,
  dtype : @precision.AutocastFloat,
) -> String {
  match dtype {
    Bfloat16 => source
    Float16 =>
      source
      .replace_all(old="#include <cuda_bf16.h>", new="#include <cuda_fp16.h>")
      .replace_all(old="__nv_bfloat16", new="__half")
      .replace_all(old="__float2bfloat16_rn", new="__float2half_rn")
      .replace_all(old="__bfloat162float", new="__half2float")
  }
}

///|
pub fn include_source(dtype : @precision.AutocastFloat) -> String {
  match dtype {
    Float16 => "#include <cuda_fp16.h>\n"
    Bfloat16 => "#include <cuda_bf16.h>\n"
  }
}

///|
/// F32 memory representation with a named low-precision rounding boundary.
pub fn round_source(value : String, dtype : @precision.AutocastFloat) -> String {
  match dtype {
    Float16 => "__half2float(__float2half_rn(\{value}))"
    Bfloat16 => "__bfloat162float(__float2bfloat16_rn(\{value}))"
  }
}

///|
pub(all) suberror LoweringError {
  InvalidSymbol
  UnsupportedPlacement
  UnsupportedGeometry
} derive(Debug, Eq)

///|
pub struct ConversionSource {
  priv text : String
  priv symbol : String
  priv representation : @format.Representation
  priv launch : ConversionLaunch
}

///|
fn valid_symbol(symbol : String) -> Bool {
  if symbol.length() < 1 || symbol.length() > 96 {
    return false
  }
  for i, c in symbol {
    if !(c is ('a'..='z' | 'A'..='Z' | '_')) && !(i > 0 && c is ('0'..='9')) {
      return false
    }
  }
  true
}

///|
pub fn ConversionSource::source(self : ConversionSource) -> String {
  self.text
}

///|
pub fn ConversionSource::symbol(self : ConversionSource) -> String {
  self.symbol
}

///|
pub fn ConversionSource::representation(
  self : ConversionSource,
) -> @format.Representation {
  self.representation
}

///|
pub fn ConversionSource::digest(self : ConversionSource) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(self.text)))
}

///|
/// Fixed logical row/scale strides specialize once at AOT generation time.
/// No per-token format switch, filesystem lookup, or runtime compilation.
pub fn decode_source(
  plan : @precision.OperandPlan,
  symbol~ : String,
) -> ConversionSource raise LoweringError {
  if !valid_symbol(symbol) {
    raise InvalidSymbol
  }
  // This kernel writes the whole tensor. A tile-sized scratch plan must not
  // accidentally authorize a model-sized conversion output allocation.
  if plan.placement() != StartupExpandBf16 || plan.role() != Weight {
    raise UnsupportedPlacement
  }
  let rep = plan.representation()
  let (rows, columns) = rep.shape()
  let geometry = launch(
    (rows.to_int64() * columns.to_int64() + 255L) / 256L,
    256,
  )
  let helper = symbol + "_load"
  let text = prelude() +
    load_helper(rep, helper) +
    (
      $|extern "C" __global__ void \{symbol}(const unsigned char *payload, const unsigned char *scales, const float *global, __nv_bfloat16 *output, int *error) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      $|  if (i >= \{rows.to_int64() * columns.to_int64()}ULL) return;
      $|  const float value = \{helper}(payload, scales, global, i / \{columns}ULL, i % \{columns}ULL);
      $|  if (!isfinite(value)) { atomicOr(error, 1); output[i] = __float2bfloat16_rn(0.0f); return; }
      $|  output[i] = __float2bfloat16_rn(fminf(3.3895313892515355e38f, fmaxf(-3.3895313892515355e38f, value)));
      $|}
    )
  { text, symbol, representation: rep, launch: geometry, }
}

///|
/// Terminal staging expression consumed with the existing physical layout.
/// It preserves the explicit decode→F32 scale→BF16 round boundary before MMA.
pub fn tile_load_source(
  physical : @precision.PhysicalOperand,
  symbol~ : String,
) -> String raise LoweringError {
  if !valid_symbol(symbol) {
    raise InvalidSymbol
  }
  if physical.plan().placement() != DecodeInOperandTile {
    raise UnsupportedPlacement
  }
  prelude() +
  load_helper(physical.plan().representation(), symbol + "_f32", checked=false) +
  (
    $|__device__ __forceinline__ __nv_bfloat16 \{symbol}(const unsigned char *payload, const unsigned char *scales, const float *global, unsigned long long row, unsigned long long column) {
    $|  const float value=\{symbol}_f32(payload, scales, global, row, column);
    $|  return __float2bfloat16_rn(fminf(3.3895313892515355e38f, fmaxf(-3.3895313892515355e38f, value)));
    $|}
  )
}

///|
fn prelude() -> String {
  (
    $|#include <cuda_bf16.h>
    $|#include <cuda_fp8.h>
    $|#include <cuda_fp16.h>
    $|#include <math.h>
    $|#include <stdint.h>
    $|#ifndef LF_PRECISION_HELPERS_V1
    $|#define LF_PRECISION_HELPERS_V1
    $|__device__ __forceinline__ unsigned lf_precision_read(const unsigned char *p, unsigned long long i, int width) { unsigned c = 0; for (int k=0;k<width;k++) c |= (unsigned)p[i+k] << (8*k); return c; }
    $|__device__ __forceinline__ float lf_precision_e4(unsigned c) { __nv_fp8_e4m3 v; v.__x = (unsigned char)c; return (float)v; }
    $|__device__ __forceinline__ float lf_precision_e5(unsigned c) { __nv_fp8_e5m2 v; v.__x = (unsigned char)c; return (float)v; }
    $|__device__ __forceinline__ float lf_precision_e2(unsigned c) { const float v[8]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f}; return (c&8U) ? -v[c&7U] : v[c&7U]; }
    $|__device__ __forceinline__ float lf_precision_ue8(unsigned c) { return c==255U ? nanf("") : c==0U ? __uint_as_float(0x00400000U) : __uint_as_float(c<<23); }
    $|#endif
    $|
  )
}

///|
fn element_decode(element : @format.Element, code : String) -> String {
  match element {
    Bf16 => "__uint_as_float((\{code})<<16)"
    Fp16 => "__half2float(__ushort_as_half((unsigned short)(\{code})))"
    F32 => "__uint_as_float(\{code})"
    E4M3 => "lf_precision_e4(\{code})"
    E5M2 => "lf_precision_e5(\{code})"
    E2M1 => "lf_precision_e2(\{code})"
    Signed8 => "(float)(signed char)(\{code})"
    Signed4 => "(float)((int)((\{code})^8U)-8)"
  }
}

///|
fn scale_index(rep : @format.Representation) -> String {
  let (_, scale_columns) = rep.scale_shape()
  match rep.grouping() {
    Absent | Tensor => "0ULL"
    Row => "row"
    Block(rows=br, columns=bc) =>
      "(row/\{br}ULL)*\{scale_columns}ULL+column/\{bc}ULL"
  }
}

///|
fn load_helper(
  rep : @format.Representation,
  symbol : String,
  checked? : Bool = true,
) -> String {
  let bits = rep.element().bits()
  let read = if bits == 4 {
    "((unsigned)payload[row*\{rep.row_bytes()}ULL+column/2ULL]>>((column%2ULL)*4))&15U"
  } else {
    "lf_precision_read(payload,row*\{rep.row_bytes()}ULL+column*\{bits/8}ULL,\{bits/8})"
  }
  let scale = match rep.scale_element() {
    Absent => ""
    value => {
      let index = scale_index(rep)
      let decode = match value {
        F32 => "__uint_as_float(lf_precision_read(scales, si*4ULL, 4))"
        E4M3 => "lf_precision_e4(scales[si])"
        Ue8M0 => "lf_precision_ue8(scales[si])"
        Absent => "1.0f"
      }
      "const unsigned long long si=\{index}; const float s=\{decode}; " +
      (if checked {
        "if (!(s>0.0f) || !isfinite(s)) return nanf(\"\"); "
      } else {
        ""
      }) +
      "value=__fmul_rn(value,s);"
    }
  }
  let global = if rep.has_global_scale() {
    (if checked {
      "if (!(*global>0.0f) || !isfinite(*global)) return nanf(\"\"); "
    } else {
      ""
    }) +
    "value=__fmul_rn(value,*global);"
  } else {
    ""
  }
  (
    $|__device__ __forceinline__ float \{symbol}(const unsigned char *payload, const unsigned char *scales, const float *global, unsigned long long row, unsigned long long column) {
    $|  const unsigned code=\{read}; float value=\{element_decode(rep.element(), "code")};
    $|  \{scale}
    $|  \{global}
    $|  return value;
    $|}
    $|
  )
}

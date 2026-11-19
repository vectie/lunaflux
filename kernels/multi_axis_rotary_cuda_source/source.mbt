///|
pub struct MultiAxisRotaryCudaSource {
  priv plan : @precision.MultiAxisRotaryPrecision
  priv symbol : String
  priv source : Bytes
  priv coefficients : Bool
}

///|
fn valid_symbol(symbol : String) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 100 &&
  symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
  symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_'))
}

///|
/// Produce BF16 cosine/sine planes from [row,T/H/W] int32 positions. Constants
/// retain their exact FP32 bits. This producer is shared by Q and K rotations.
pub fn coefficients_source(
  plan : @precision.MultiAxisRotaryPrecision,
  symbol~ : String,
) -> MultiAxisRotaryCudaSource raise {
  guard valid_symbol(symbol) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (_, _, rotary, rows, positions) = plan.geometry()
  let pairs = rotary / 2
  let axes = plan.axes().map(axis => axis.to_string()).join(",")
  let bits = plan
    .inverse_frequencies()
    .map(f => f.reinterpret_as_uint().to_string() + "U")
    .join(",")
  let angle = @arithmetic.scalar_expression(plan.expression(Angle), [
    "(float)position",
    "__uint_as_float(\{symbol}_frequency[pair])",
  ])
  let cosine = @arithmetic.scalar_expression(plan.expression(Cosine), ["angle"])
  let sine = @arithmetic.scalar_expression(plan.expression(Sine), ["angle"])
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|static __device__ __constant__ int \{symbol}_axis[\{pairs}]={\{axes}};
    $|static __device__ __constant__ unsigned int \{symbol}_frequency[\{pairs}]={\{bits}};
    $|extern "C" __global__ void \{symbol}(const int *counts, const int *positions, __nv_bfloat16 *cosine, __nv_bfloat16 *sine) {
    $|  const int row=blockIdx.x;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]) return;
    $|  for(int pair=threadIdx.x;pair<\{pairs};pair+=blockDim.x) {
    $|    const int position=positions[(unsigned long long)row*3ULL+\{symbol}_axis[pair]];
    $|    if(position<0||position>=\{positions}) continue;
    $|    const float angle=\{angle};
    $|    const unsigned long long index=(unsigned long long)row*\{pairs}ULL+pair;
    $|    cosine[index]=\{cosine}; sine[index]=\{sine};
    $|  }
    $|}
  { plan, symbol, source: @utf8.encode(text), coefficients: true, }
}

///|
/// Rotate a prefix using split halves. Input and output may alias; tail words
/// are preserved bit-for-bit. Coefficients are packed [row,rotary_width/2].
pub fn rotary_source(
  plan : @precision.MultiAxisRotaryPrecision,
  symbol~ : String,
) -> MultiAxisRotaryCudaSource raise {
  let (heads, width, rotary, rows, _) = plan.geometry()
  guard valid_symbol(symbol) && heads <= 65535 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let pairs = rotary / 2
  let real = @arithmetic.scalar_expression(plan.expression(Real), [
    "left", "right", "c", "s",
  ])
  let imaginary = @arithmetic.scalar_expression(plan.expression(Imaginary), [
    "left", "right", "c", "s",
  ])
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine, __nv_bfloat16 *output) {
    $|  const int row=blockIdx.x, head=blockIdx.y;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]||head>=\{heads}) return;
    $|  const unsigned long long base=((unsigned long long)row*\{heads}ULL+head)*\{width}ULL;
    $|  for(int pair=threadIdx.x;pair<\{pairs};pair+=blockDim.x) {
    $|    const unsigned long long index=base+pair, coefficient=(unsigned long long)row*\{pairs}ULL+pair;
    $|    const __nv_bfloat16 left=input[index],right=input[index+\{pairs}ULL],c=cosine[coefficient],s=sine[coefficient];
    $|    output[index]=\{real}; output[index+\{pairs}ULL]=\{imaginary};
    $|  }
    $|  for(int column=\{rotary}+threadIdx.x;column<\{width};column+=blockDim.x)
    $|    reinterpret_cast<uint16_t *>(output)[base+column]=reinterpret_cast<const uint16_t *>(input)[base+column];
    $|}
  { plan, symbol, source: @utf8.encode(text), coefficients: false, }
}

///|
pub fn MultiAxisRotaryCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn MultiAxisRotaryCudaSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn MultiAxisRotaryCudaSource::plan(
  self : Self,
) -> @precision.MultiAxisRotaryPrecision {
  self.plan
}

///|
pub fn MultiAxisRotaryCudaSource::launch(self : Self) -> (Int32, Int32, Int32) {
  let (heads, _, _, rows, _) = self.plan.geometry()
  (rows, if self.coefficients { 1 } else { heads }, 128)
}

///|
pub fn MultiAxisRotaryCudaSource::parameter_count(self : Self) -> Int32 {
  if self.coefficients {
    4
  } else {
    5
  }
}

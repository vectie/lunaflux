///|
/// Standalone compact rows, without redundant query rotation/allocation.
pub struct RotaryCacheCudaSource {
  priv plan : @precision.RotaryCachePrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn rotary_cache_source(
  plan : @precision.RotaryCachePrecision,
  symbol_prefix~ : String,
) -> RotaryCacheCudaSource raise {
  guard valid_symbol(symbol_prefix) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, rotary, rows, _) = plan.geometry()
  let prefix = width - rotary
  let symbols = [symbol_prefix + "_rotary", symbol_prefix + "_simulate"]
  let rotation = rotary_tensor_source(plan.rotary(), symbol=symbols[0])
  let (block, floor, scale) = plan.simulation()
  let simulation = cache_simulation_source(
    symbols[1],
    rows,
    heads,
    width,
    prefix,
    block,
    floor,
    scale,
  )
  let source = @utf8.encode(
    (
      $|#include <cuda.h>
      $|#include <cuda_bf16.h>
      $|#include <cuda_fp8.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|\{rotation}
      $|\{simulation}
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn RotaryCacheCudaSource::plan(
  self : Self,
) -> @precision.RotaryCachePrecision {
  self.plan
}

///|
pub fn RotaryCacheCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RotaryCacheCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RotaryCacheCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, width, rotary, rows, _) = self.plan.geometry()
  let (block, _, _) = self.plan.simulation()
  if stage == 0 {
    (rows, heads, 256)
  } else {
    (rows, Int::max(1, heads * ((width - rotary) / block)), block)
  }
}

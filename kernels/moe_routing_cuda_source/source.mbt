///|
/// Terminal lowering of numeric routing IR. Source is compiled offline only.
pub struct MoeRoutingCudaSource {
  priv plan : @precision.MoeRoutingPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
  priv launches : ReadOnlyArray[(Int, Int)]
}

///|
pub fn routing_source(
  plan : @precision.MoeRoutingPrecision,
  symbol~ : String,
) -> MoeRoutingCudaSource raise {
  let (rows, _, experts, selected) = plan.geometry()
  let (groups, group_topk, selected_groups) = plan.grouping()
  let (selection_source, selection_symbols, selection_launches) = match
    plan.selection() {
    GroupedTopK => {
      let choose = @grouped.grouped_choice_and_selected_weight_f32_source(
        routed_expert_count=experts,
        group_count=groups,
        group_score_topk=group_topk,
        selected_group_count=selected_groups,
        experts_per_token=selected,
        normalize=plan.normalize(),
        scaling=plan.scaling(),
        maximum_rows=rows,
        function_symbol=symbol + "_choose",
        selected_sum_law=plan.selected_sum_law(),
      )
      (@utf8.decode(choose.source_bytes()), [symbol + "_choose"], [(rows, 1)])
    }
    TokenHash(vocabulary_size~) => {
      let lookup = @hash.token_hash_lookup_i32_source(
        vocabulary_size~,
        experts_per_token=selected,
        routed_expert_count=experts,
        maximum_rows=rows,
        function_symbol=symbol + "_lookup",
      )
      (
        @utf8.decode(lookup.source_bytes()) +
        hash_weight_source(plan, symbol + "_gather"),
        [symbol + "_lookup", symbol + "_gather"],
        [(rows, 256), (rows, 1)],
      )
    }
  }
  let cells = rows.to_int64() * experts.to_int64()
  let grid = (cells + 255L) / 256L
  guard grid <= 2147483647L else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  let weight_type = if plan.weight().representation().format() == Bf16 {
    "__nv_bfloat16"
  } else {
    "float"
  }
  let weight_load = if weight_type == "float" {
    "weight[base+k]"
  } else {
    "__bfloat162float(weight[base+k])"
  }
  let threshold = match plan.softplus_law() {
    StableFloat32 => "0.0f"
    ThresholdedFloat32(threshold~) =>
      @literal.canonical_cuda_float_literal(threshold)
  }
  let score = @arithmetic.scalar_expression(plan.score_expression(), [
    "x", threshold,
  ])
  let score_arguments = if plan.selection() is GroupedTopK {
    "const float *bias, float *scores, float *choice"
  } else {
    "float *scores"
  }
  let correction = if plan.selection() is GroupedTopK {
    "choice[cell]=__fadd_rn(score,bias[cell%" + experts.to_string() + "ULL]);"
  } else {
    ""
  }
  let (projection, projection_launch) = projection_source(
    plan, symbol, weight_type, weight_load,
  )
  let headers =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
  let score_source =
    $|extern "C" __global__ void \{symbol}_score(const int *counts, const float *logits, \{score_arguments}) {
    $|  const int live=counts[3]; if(live<=0||live>\{rows}) return;
    $|  const unsigned long long cell=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    $|  if(cell>=(unsigned long long)live*\{experts}ULL) return;
    $|  const float x=logits[cell]; const float score=\{score};
    $|  scores[cell]=score; \{correction}
    $|}
  {
    plan,
    source: @utf8.encode(
      headers +
      "\n" +
      projection +
      "\n" +
      score_source +
      "\n" +
      selection_source,
    ),
    symbols: ReadOnlyArray::from_array(
      [symbol + "_project", symbol + "_score"] + selection_symbols,
    ),
    launches: ReadOnlyArray::from_array(
      [projection_launch, (grid.to_int(), 256)] + selection_launches,
    ),
  }
}

///|
pub fn MoeRoutingCudaSource::plan(
  self : Self,
) -> @precision.MoeRoutingPrecision {
  self.plan
}

///|
pub fn MoeRoutingCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn MoeRoutingCudaSource::symbols(self : Self) -> ReadOnlyArray[String] {
  self.symbols
}

///|
pub fn MoeRoutingCudaSource::launches(self : Self) -> ReadOnlyArray[(Int, Int)] {
  self.launches
}

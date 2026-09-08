///|
/// A selected variant owns its complete launch rule. Never combine its
/// function with another variant's tile width or shared-memory requirement.
priv enum PagedBucketLaunchPhase {
  BaselineLaunch
  DecodeLaunch
  WidePrefillLaunch
  SplitPrefillLaunch
}

///|
/// Pure selection; an absent specialization keeps the selected step's exact
/// launch, rather than borrowing geometry from the baseline kernel.
fn paged_phase_bucket_rule(
  contract : PagedGraphLaunchContract,
  phase : PagedBucketLaunchPhase,
  index : Int,
) -> PagedBucketStepLaunch {
  let rules = match phase {
    BaselineLaunch => Some(contract.step_launches)
    DecodeLaunch => Some(contract.decode_step_launches)
    WidePrefillLaunch => contract.prefill_wide_step_launches
    SplitPrefillLaunch => contract.prefill_split_step_launches
  }
  match rules {
    Some(rules) => rules.get(index).unwrap_or(FixedBucketLaunch)
    None => FixedBucketLaunch
  }
}

///|
/// Derive a separate startup value for the wide variant. Projection and
/// ingress rules are unchanged; attention carries the selected variant's
/// complete geometry and semantic query tile. The baseline stays immutable.
fn prepare_wide_prefill_bucket_contract(
  blueprint : PagedKvExecutionBlueprint,
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  baseline_rules : FixedArray[PagedBucketStepLaunch],
  selected : ReadOnlyArray[PreparedPagedGraphStep]?,
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> FixedArray[PagedBucketStepLaunch]? raise DeviceStepError {
  guard selected is Some(steps) else { return None }
  if steps.length() != baseline.length() ||
    baseline_rules.length() != baseline.length() {
    raise InvalidExecutor(GraphBucket)
  }
  for index, step in steps {
    if step.operation_id != baseline[index].operation_id {
      raise InvalidExecutor(GraphBucket)
    }
  }
  let rules = FixedArray::makei(baseline_rules.length(), i => baseline_rules[i])
  for span in spans {
    guard span.wide_query_prefill() is Some(variant) &&
      blueprint.steps().get(span.attention_index) is Some(attention) else {
      raise InvalidExecutor(GraphBucket)
    }
    let index = qwen_bucket_step_index(steps, attention.operation_id())
    rules[index] = QueryTileCappedGridX(variant.launch, variant.query_tile_rows)
  }
  Some(rules)
}

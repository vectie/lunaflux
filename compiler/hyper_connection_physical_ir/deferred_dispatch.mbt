///|
/// Offline-selected row domains. Measurements select these records outside
/// the compiler; this immutable plan only checks legality and exhaustive
/// coverage. No profiler, model-family condition, or request-path tuning.
pub struct DeferredPostPreDispatch {
  priv buckets : ReadOnlyArray[(Int32, Int32)]
  priv workgroups : Int32
}

///|
pub fn DeferredPostPreDispatch::new(
  producer : @precision.HyperConnectionPrecision,
  consumer : @precision.HyperConnectionPrecision,
  buckets : ArrayView[(Int32, Int32)],
  workgroup_budget~ : Int32,
) -> Self raise @precision.PrecisionPlanError {
  let (_, _, rows) = consumer.geometry()
  let mut last = 0
  let mut groups = 0
  for (end, width) in buckets {
    guard end > last && end <= rows else { raise IllegalPlacement }
    let plan = DeferredPostPrePlan::new(
      producer,
      consumer,
      controls_per_group=width,
      workgroup_budget~,
    )
    groups = Int32::max(groups, plan.workgroups())
    last = end
  }
  guard last == rows else { raise IllegalPlacement }
  { buckets: ReadOnlyArray::from_array(buckets), workgroups: groups, }
}

///|
pub fn DeferredPostPreDispatch::buckets(
  self : Self,
) -> ReadOnlyArray[(Int32, Int32)] {
  self.buckets
}

///|
pub fn DeferredPostPreDispatch::workgroups(self : Self) -> Int32 {
  self.workgroups
}

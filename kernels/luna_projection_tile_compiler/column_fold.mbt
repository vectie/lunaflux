///|
/// Physical loop IR for a map of independent ordered dot-product folds.
/// Each window keeps a bounded number of accumulators per consumer alive.
/// Inside that window the reduction axis is outermost: its input operand is
/// shared by the independent column folds, without reassociating any fold.
/// Consumers without a column still participate in workgroup effects.
pub struct ProjectionColumnFoldPlan {
  priv output_tiles : Int
  priv consumer_groups : Int
  priv live_columns : Int
  priv rounds : Int
} derive(Debug, Eq)

///|
pub extend ProjectionColumnFoldPlan with @debug.Debug::{to_repr}

///|
pub extend ProjectionColumnFoldPlan with Eq::{equal, not_equal}

///|
pub fn ProjectionColumnFoldPlan::new(
  output_tiles~ : Int,
  consumer_groups~ : Int,
  maximum_live_accumulators~ : Int,
) -> ProjectionColumnFoldPlan? {
  guard output_tiles > 0 && consumer_groups > 0 && maximum_live_accumulators > 0 else {
    return None
  }
  let rounds = (output_tiles - 1) / consumer_groups + 1
  Some({
    output_tiles,
    consumer_groups,
    live_columns: if rounds < maximum_live_accumulators {
      rounds
    } else {
      maximum_live_accumulators
    },
    rounds,
  })
}

///|
/// (consumer groups, total rounds, live columns per consumer, windows).
pub fn ProjectionColumnFoldPlan::geometry(
  self : ProjectionColumnFoldPlan,
) -> (Int, Int, Int, Int) {
  (
    self.consumer_groups,
    self.rounds,
    self.live_columns,
    (self.rounds - 1) / self.live_columns + 1,
  )
}

///|
pub fn ProjectionColumnFoldPlan::column_at(
  self : ProjectionColumnFoldPlan,
  window : Int,
  slot : Int,
  consumer : Int,
) -> Int? {
  let (_, _, _, windows) = self.geometry()
  guard window >= 0 &&
    window < windows &&
    slot >= 0 &&
    slot < self.live_columns &&
    consumer >= 0 &&
    consumer < self.consumer_groups else {
    return None
  }
  let round_base = window * self.live_columns
  guard slot < self.rounds - round_base else { return None }
  let base = (round_base + slot) * self.consumer_groups
  if consumer < self.output_tiles - base {
    Some(base + consumer)
  } else {
    None
  }
}

///|
pub fn ProjectionColumnFoldPlan::canonical_bytes(
  self : ProjectionColumnFoldPlan,
) -> Bytes {
  @utf8.encode(
    "luna-projection-column-fold-v1\n" +
    "output_tiles=\{self.output_tiles}\n" +
    "consumer_groups=\{self.consumer_groups}\n" +
    "live_columns=\{self.live_columns}\n" +
    "order=window,reduction,column\n" +
    "input_scope=reduction\n" +
    "accumulator_scope=window\n" +
    "effects=all-consumers\n" +
    "reduction=ordered\n",
  )
}

///|
fn ProjectionTileSchedule::column_fold_axis(
  self : ProjectionTileSchedule,
  axis : ProjectionIterationAxis,
) -> ProjectionLoopCombinator? {
  for item in self.loops {
    if item.axis == axis {
      return Some(item)
    }
  }
  None
}

///|
/// Refine a scheduled projection into a bounded physical operand-reuse plan.
/// An ingress epilogue consumes a complete head; its projection domain is the
/// head-component map, not the concatenation of all output heads. Hardware
/// supplies group count and resource budget, never the model builder.
pub fn ProjectionTileSchedule::column_fold_plan(
  self : ProjectionTileSchedule,
  consumer_groups~ : Int,
  maximum_live_accumulators~ : Int,
) -> ProjectionColumnFoldPlan? {
  guard self.row_tail_policy == ZeroPaddedMatrixRowTail else { return None }
  let column = self.column_fold_axis(ProjectionOutputColumnAxis)
  guard column is Some(column) else { return None }
  let extent = if self.fuse_attention_ingress_epilogue {
    guard self.column_fold_axis(ProjectionHeadComponentAxis) is Some(head) else {
      return None
    }
    head.extent
  } else {
    column.extent
  }
  guard extent > 0 && column.tile > 0 else { return None }
  ProjectionColumnFoldPlan::new(
    output_tiles=(extent - 1) / column.tile + 1,
    consumer_groups~,
    maximum_live_accumulators~,
  )
}

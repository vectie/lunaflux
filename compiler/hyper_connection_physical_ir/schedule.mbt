///|
pub(all) enum ControlSchedule {
  Ordered(OrderedControlPlan)
  Tree(ParallelControlPlan)
}

///|
pub fn ControlSchedule::fuses_normalization(self : Self) -> Bool {
  match self {
    Ordered(_) => false
    Tree(p) => p.fuses_normalization()
  }
}

///|
pub fn ControlSchedule::workers(self : Self) -> Int {
  match self {
    Ordered(p) => p.workers()
    Tree(p) => p.workers()
  }
}

///|
pub fn ControlSchedule::workspace_bytes(self : Self) -> Int64 {
  match self {
    Ordered(p) => p.workspace_bytes()
    Tree(p) => p.workspace_bytes()
  }
}

///|
pub fn ControlSchedule::workgroups(
  self : Self,
  stage : Int,
) -> Int raise @precision.PrecisionPlanError {
  match self {
    Ordered(p) => p.workgroups(stage)
    Tree(p) => p.workgroups(stage)
  }
}

///|
/// Bounded ahead-of-use operand values. Consuming a window remains ordered in
/// K; loading the next value into a temporary does not advance the accumulator.
pub struct ProjectionOperandWindow {
  priv extent : Int
  priv quantum : Int
  priv slots : Int
} derive(Debug, Eq)

///|
pub extend ProjectionOperandWindow with Eq::{equal, not_equal}

///|
pub extend ProjectionOperandWindow with @debug.Debug::{to_repr}

///|
pub fn ProjectionOperandWindow::new(
  extent~ : Int,
  quantum~ : Int,
  slots~ : Int,
) -> ProjectionOperandWindow? {
  guard extent > 0 &&
    quantum > 0 &&
    extent % quantum == 0 &&
    slots > 0 &&
    slots <= extent / quantum else {
    return None
  }
  Some({ extent, quantum, slots, })
}

///|
pub fn ProjectionOperandWindow::geometry(
  self : ProjectionOperandWindow,
) -> (Int, Int, Int, Int) {
  (self.extent, self.quantum, self.slots, self.quantum * self.slots)
}

///|
/// Same pure address relation is used for initial fill, current consumption,
/// and next-window prefetch. The last partial window never issues an extra load.
pub fn ProjectionOperandWindow::read_at(
  self : ProjectionOperandWindow,
  window~ : Int,
  slot~ : Int,
) -> Int? {
  guard window >= 0 && slot >= 0 && slot < self.slots else { return None }
  let fragment = window.to_int64() * self.slots.to_int64() + slot.to_int64()
  guard fragment < (self.extent / self.quantum).to_int64() else { return None }
  Some(fragment.to_int() * self.quantum)
}

///|
pub fn ProjectionSelectedRowGather::operand_window(
  self : ProjectionSelectedRowGather,
  input_width : Int,
) -> ProjectionOperandWindow? {
  ProjectionOperandWindow::new(
    extent=input_width,
    quantum=self.reduction,
    slots=self.prefetch_fragments,
  )
}

///|
/// A non-pipelined mapped fold still has physical shared operand identities and
/// an ordered accumulator. Resident selection changes strip extent, not MMA
/// semantics. Device-specific fragment lane maps remain in backend lowering.
pub fn ProjectionTileSchedule::mapped_fragment_program(
  self : ProjectionTileSchedule,
  input_width : Int,
) -> @physical_tile_ir.FragmentProgram? {
  for combinator in self.loops {
    if combinator.axis == ProjectionReductionAxis {
      guard combinator.kind == ProjectionSequentialFold &&
        combinator.extent == input_width &&
        combinator.tile > 0 &&
        input_width > 0 &&
        input_width % combinator.tile == 0 else {
        return None
      }
      let extent = if self.selected_input_row_resident() {
        input_width
      } else {
        combinator.tile
      }
      return @physical_tile_ir.FragmentProgram::new(
        rows=1,
        columns=1,
        extent~,
        quantum=combinator.tile,
        slots=1,
      )
    }
  }
  None
}

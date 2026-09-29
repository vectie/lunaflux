///|
/// Abstract matrix operands. Instruction widths and lane mappings belong to
/// the backend; neither changes the ordered reduction described here.
pub(all) enum FragmentOperand {
  LeftOperand
  RightOperand
} derive(Eq, Debug)

///|
pub extend FragmentOperand with Eq::{equal, not_equal}

///|
pub extend FragmentOperand with @debug.Debug::{to_repr}

///|
pub(all) enum FragmentAction {
  ReadCurrent
  ReadNextIfPresent
  AccumulateCurrent
} derive(Eq, Debug)

///|
pub extend FragmentAction with Eq::{equal, not_equal}

///|
pub extend FragmentAction with @debug.Debug::{to_repr}

///|
/// One product of ordered folds over a published operand tile. Loads are SSA
/// values: right column c has ID c, left row r has ID columns+r. Products refer
/// to those values, rather than hiding repeated loads in arithmetic strings.
/// Construction is constant-space even for large representable domains.
pub struct FragmentProgram {
  priv rows : Int
  priv columns : Int
  priv extent : Int
  priv quantum : Int
  priv slots : Int
} derive(Eq, Debug)

///|
pub extend FragmentProgram with Eq::{equal, not_equal}

///|
pub extend FragmentProgram with @debug.Debug::{to_repr}

///|
pub fn FragmentProgram::new(
  rows~ : Int,
  columns~ : Int,
  extent~ : Int,
  quantum~ : Int,
  slots~ : Int,
) -> FragmentProgram? {
  guard rows > 0 &&
    columns > 0 &&
    extent > 0 &&
    quantum > 0 &&
    extent % quantum == 0 &&
    slots >= 1 &&
    slots <= 2 &&
    slots <= extent / quantum &&
    rows.to_int64() + columns.to_int64() <= 2147483647L &&
    rows.to_int64() * columns.to_int64() <= 2147483647L &&
    (rows.to_int64() + columns.to_int64()) * slots.to_int64() <= 2147483647L else {
    return None
  }
  Some({ rows, columns, extent, quantum, slots, })
}

///|
pub fn FragmentProgram::geometry(
  self : FragmentProgram,
) -> (Int, Int, Int, Int, Int) {
  (self.rows, self.columns, self.extent, self.quantum, self.slots)
}

///|
pub fn FragmentProgram::load_count(self : FragmentProgram) -> Int {
  self.rows + self.columns
}

///|
/// Logical shared-tile access for one SSA fragment value.
pub fn FragmentProgram::load(
  self : FragmentProgram,
  value : Int,
) -> (FragmentOperand, Int)? {
  if value < 0 || value >= self.load_count() {
    None
  } else if value < self.columns {
    Some((RightOperand, value))
  } else {
    Some((LeftOperand, value - self.columns))
  }
}

///|
pub fn FragmentProgram::product_count(self : FragmentProgram) -> Int {
  self.rows * self.columns
}

///|
/// (left value, right value, accumulator). The accumulator is loop-carried;
/// independent products can share reads, but no pass may reassociate its fold.
pub fn FragmentProgram::product(
  self : FragmentProgram,
  output : Int,
) -> (Int, Int, Int)? {
  if output < 0 || output >= self.product_count() {
    None
  } else {
    Some((self.columns + output / self.columns, output % self.columns, output))
  }
}

///|
pub fn FragmentProgram::primes_operands(self : FragmentProgram) -> Bool {
  self.slots > 1
}

///|
pub fn FragmentProgram::iteration(
  self : FragmentProgram,
) -> ReadOnlyArray[FragmentAction] {
  if self.primes_operands() {
    [ReadNextIfPresent, AccumulateCurrent]
  } else {
    [ReadCurrent, AccumulateCurrent]
  }
}

///|
/// A read's reduction offset and destination register slot. Next reads are
/// absent at the tail, never fabricated out-of-range accesses.
pub fn FragmentProgram::read_at(
  self : FragmentProgram,
  iteration : Int,
  next~ : Bool,
) -> (Int, Int)? {
  let count = self.extent / self.quantum
  guard iteration >= 0 && iteration < count else { return None }
  if next && (self.slots == 1 || iteration == count - 1) {
    return None
  }
  let index = iteration + (if next { 1 } else { 0 })
  Some((index * self.quantum, index % self.slots))
}

///|
pub fn FragmentProgram::canonical(self : FragmentProgram) -> String {
  "tile-fragment-product-v1;rows=\{self.rows};columns=\{self.columns};extent=\{self.extent};quantum=\{self.quantum};slots=\{self.slots};order=ordered;loads=right-then-left;products=row-major"
}

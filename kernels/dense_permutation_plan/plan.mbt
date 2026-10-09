///|
pub(all) suberror PermutationError {
  InvalidShape
  InvalidOrder
  ElementLimitExceeded
  IndexOutOfBounds
}

///|
/// One immutable term in an output-linear to input-linear index expression.
/// Consecutive axes contiguous in both layouts are collapsed at preparation.
pub struct IndexTerm {
  extent : Int64
  output_stride : Int64
  input_stride : Int64
}

///|
/// Dense contiguous tensor permutation. Shape splitting/merging is semantic:
/// e.g. video [B,C,T,H,W] is split into [B,C,t,pt,h,ph,w,pw] before permutation.
/// There are no device names, launch dimensions, or tensor payloads here.
pub struct Permutation {
  priv source_shape : ReadOnlyArray[Int]
  priv output_order : ReadOnlyArray[Int]
  priv terms : ReadOnlyArray[IndexTerm]
  priv elements : Int64
}

///|
/// Build a bijective permutation under an explicit element ceiling.
/// `output_order[j]` is the source axis occupying output axis j.
pub fn Permutation::new(
  source_shape : ArrayView[Int],
  output_order : ArrayView[Int],
  max_elements~ : Int64,
) -> Self raise PermutationError {
  let rank = source_shape.length()
  guard rank > 0 && rank <= 8 else { raise InvalidShape }
  guard output_order.length() == rank else { raise InvalidOrder }
  guard max_elements > 0L else { raise ElementLimitExceeded }
  let strides = Array::make(rank, 0L)
  let elements = for axis = rank - 1, stride = 1L; axis >= 0; {
    let extent = source_shape[axis].to_int64()
    guard extent > 0L else { raise InvalidShape }
    guard stride <= max_elements / extent else { raise ElementLimitExceeded }
    strides[axis] = stride
    continue axis - 1, stride * extent
  } nobreak {
    stride
  }
  let seen = Array::make(rank, false)
  for axis in output_order {
    guard axis >= 0 && axis < rank && !seen[axis] else { raise InvalidOrder }
    seen[axis] = true
  }
  let terms : Array[IndexTerm] = []
  for axis in output_order; output_stride = elements {
    let extent = source_shape[axis].to_int64()
    let next_stride = output_stride / extent
    if extent != 1L {
      let item : IndexTerm = {
        extent,
        output_stride: next_stride,
        input_stride: strides[axis],
      }
      if terms.length() > 0 {
        let previous = terms[terms.length() - 1]
        if previous.input_stride == item.input_stride * item.extent {
          terms[terms.length() - 1] = {
            extent: previous.extent * item.extent,
            output_stride: item.output_stride,
            input_stride: item.input_stride,
          }
        } else {
          terms.push(item)
        }
      } else {
        terms.push(item)
      }
    }
    continue next_stride
  }
  {
    source_shape: ReadOnlyArray::from_array(source_shape.to_owned()),
    output_order: ReadOnlyArray::from_array(output_order.to_owned()),
    terms: ReadOnlyArray::from_array(terms),
    elements,
  }
}

///|
pub fn Permutation::element_count(self : Self) -> Int64 {
  self.elements
}

///|
pub fn Permutation::terms(self : Self) -> ReadOnlyArray[IndexTerm] {
  self.terms
}

///|
pub fn Permutation::is_identity(self : Self) -> Bool {
  self.elements == 1L ||
  (self.terms.length() == 1 && self.terms[0].input_stride == 1L)
}

///|
/// Pure scalar interpretation, useful for host staging and numerical oracles.
/// Device lowering consumes terms directly, without per-element validation.
pub fn Permutation::source_index(
  self : Self,
  output_index : Int64,
) -> Int64 raise PermutationError {
  guard output_index >= 0L && output_index < self.elements else {
    raise IndexOutOfBounds
  }
  for term in self.terms; source = 0L {
    continue source +
      output_index / term.output_stride % term.extent * term.input_stride
  } nobreak {
    source
  }
}

///|
/// Inversion preserves the same finite element set and reconstructs a pure
/// plan; it performs no payload permutation or allocation of tensor storage.
pub fn Permutation::inverse(self : Self) -> Self {
  let shape = Array::make(self.source_shape.length(), 0)
  let order = Array::make(self.source_shape.length(), 0)
  for output_axis, input_axis in self.output_order {
    shape[output_axis] = self.source_shape[input_axis]
    order[input_axis] = output_axis
  }
  try! Permutation::new(shape, order, max_elements=self.elements)
}

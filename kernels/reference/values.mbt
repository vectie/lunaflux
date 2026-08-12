///|
/// Immutable, finite rank-2 activation owned by the reference package.
pub struct Activation {
  rows : Int
  columns : Int
  values : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
/// Immutable, finite rank-2 model tensor owned by the reference package.
pub struct Tensor {
  rows : Int
  columns : Int
  values : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
fn copy_finite_values(
  values : ArrayView[Float],
  role : ReferenceValueRole,
) -> ReadOnlyArray[Float] raise ReferenceKernelError {
  let copied : Array[Float] = []
  for value in values {
    if value.is_nan() || value.is_inf() {
      raise NonFinite(role)
    }
    copied.push(value)
  }
  ReadOnlyArray::from_array(copied)
}

///|
/// Copies a finite row-major matrix after validating its rank-2 shape.
pub fn Activation::new(
  rows~ : Int,
  columns~ : Int,
  values~ : ArrayView[Float],
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  let elements = checked_element_count(rows, columns, limits)
  if values.length() != elements {
    raise InvalidShape(ElementCount)
  }
  { rows, columns, values: copy_finite_values(values, ActivationValue) }
}

///|
/// Copies a finite row-major matrix after validating its rank-2 shape.
pub fn Tensor::new(
  rows~ : Int,
  columns~ : Int,
  values~ : ArrayView[Float],
  limits~ : ReferenceLimits,
) -> Tensor raise ReferenceKernelError {
  let elements = checked_element_count(rows, columns, limits)
  if values.length() != elements {
    raise InvalidShape(ElementCount)
  }
  { rows, columns, values: copy_finite_values(values, TensorValue) }
}

///|
pub fn Activation::rows(self : Activation) -> Int {
  self.rows
}

///|
pub fn Activation::columns(self : Activation) -> Int {
  self.columns
}

///|
pub fn Activation::values(self : Activation) -> ReadOnlyArray[Float] {
  self.values
}

///|
pub fn Activation::at(
  self : Activation,
  row~ : Int,
  column~ : Int,
) -> Float raise ReferenceKernelError {
  if row < 0 || row >= self.rows || column < 0 || column >= self.columns {
    raise InvalidIndex
  }
  self.values[row * self.columns + column]
}

///|
pub fn Tensor::rows(self : Tensor) -> Int {
  self.rows
}

///|
pub fn Tensor::columns(self : Tensor) -> Int {
  self.columns
}

///|
pub fn Tensor::values(self : Tensor) -> ReadOnlyArray[Float] {
  self.values
}

///|
pub fn Tensor::at(
  self : Tensor,
  row~ : Int,
  column~ : Int,
) -> Float raise ReferenceKernelError {
  if row < 0 || row >= self.rows || column < 0 || column >= self.columns {
    raise InvalidIndex
  }
  self.values[row * self.columns + column]
}

///|
fn activation_from_owned(
  rows : Int,
  columns : Int,
  values : Array[Float],
  limits : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  let elements = checked_element_count(rows, columns, limits)
  if values.length() != elements {
    raise InvalidShape(ElementCount)
  }
  for value in values {
    if value.is_nan() || value.is_inf() {
      raise NonFinite(ComputedValue)
    }
  }
  { rows, columns, values: ReadOnlyArray::from_array(values) }
}

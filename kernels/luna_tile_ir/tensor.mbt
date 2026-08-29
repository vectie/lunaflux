///|
pub struct LunaTileTensorView {
  priv id : LunaTileTensorId
  priv dtype : LunaTileDType
  priv address_space : LunaTileAddressSpace
  priv shape : ReadOnlyArray[Int64]
  priv strides_bytes : ReadOnlyArray[Int64]
  priv offset_bytes : Int64
  priv alignment_bytes : Int64
} derive(Debug, Eq)

///|
pub fn LunaTileTensorView::new(
  id~ : LunaTileTensorId,
  dtype~ : LunaTileDType,
  address_space~ : LunaTileAddressSpace,
  shape~ : ArrayView[Int64],
  strides_bytes~ : ArrayView[Int64],
  offset_bytes~ : Int64,
  alignment_bytes~ : Int64,
) -> LunaTileTensorView {
  {
    id,
    dtype,
    address_space,
    shape: ReadOnlyArray::from_array(shape),
    strides_bytes: ReadOnlyArray::from_array(strides_bytes),
    offset_bytes,
    alignment_bytes,
  }
}

///|
pub fn LunaTileTensorView::id(self : LunaTileTensorView) -> LunaTileTensorId {
  self.id
}

///|
pub fn LunaTileTensorView::dtype(self : LunaTileTensorView) -> LunaTileDType {
  self.dtype
}

///|
pub fn LunaTileTensorView::address_space(
  self : LunaTileTensorView,
) -> LunaTileAddressSpace {
  self.address_space
}

///|
pub fn LunaTileTensorView::shape(
  self : LunaTileTensorView,
) -> ReadOnlyArray[Int64] {
  self.shape
}

///|
pub fn LunaTileTensorView::strides_bytes(
  self : LunaTileTensorView,
) -> ReadOnlyArray[Int64] {
  self.strides_bytes
}

///|
pub fn LunaTileTensorView::offset_bytes(self : LunaTileTensorView) -> Int64 {
  self.offset_bytes
}

///|
pub fn LunaTileTensorView::alignment_bytes(self : LunaTileTensorView) -> Int64 {
  self.alignment_bytes
}

///|
pub struct LunaTileAffineTerm {
  priv loop_dimension : Int
  priv coefficient : Int64
} derive(Debug, Eq)

///|
pub fn LunaTileAffineTerm::new(
  loop_dimension~ : Int,
  coefficient~ : Int64,
) -> LunaTileAffineTerm {
  { loop_dimension, coefficient }
}

///|
pub struct LunaTileAffineExpr {
  priv constant : Int64
  priv terms : ReadOnlyArray[LunaTileAffineTerm]
} derive(Debug, Eq)

///|
pub fn LunaTileAffineExpr::new(
  constant~ : Int64,
  terms~ : ArrayView[LunaTileAffineTerm],
) -> LunaTileAffineExpr {
  { constant, terms: ReadOnlyArray::from_array(terms) }
}

///|
pub fn LunaTileAffineExpr::constant(self : LunaTileAffineExpr) -> Int64 {
  self.constant
}

///|
pub fn LunaTileAffineExpr::terms(
  self : LunaTileAffineExpr,
) -> ReadOnlyArray[LunaTileAffineTerm] {
  self.terms
}

///|
pub struct LunaTileRegion {
  priv tensor : LunaTileTensorId
  priv origins : ReadOnlyArray[LunaTileAffineExpr]
  priv extents : ReadOnlyArray[Int64]
} derive(Debug, Eq)

///|
pub fn LunaTileRegion::new(
  tensor~ : LunaTileTensorId,
  origins~ : ArrayView[LunaTileAffineExpr],
  extents~ : ArrayView[Int64],
) -> LunaTileRegion {
  {
    tensor,
    origins: ReadOnlyArray::from_array(origins),
    extents: ReadOnlyArray::from_array(extents),
  }
}

///|
pub fn LunaTileRegion::tensor(self : LunaTileRegion) -> LunaTileTensorId {
  self.tensor
}

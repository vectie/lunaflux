///|
pub fn f32_residual_add_reference(
  residual : ArrayView[Float],
  value : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointAudioAmpCudaSourceError {
  if residual.length() == 0 || residual.length() != value.length() {
    raise InvalidReferenceShape
  }
  ReadOnlyArray::from_array(
    Array::makei(residual.length(), index => residual[index] + value[index]),
  )
}

///|
pub fn f32_three_way_average_reference(
  block0 : ArrayView[Float],
  block1 : ArrayView[Float],
  block2 : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointAudioAmpCudaSourceError {
  if block0.length() == 0 ||
    block0.length() != block1.length() ||
    block0.length() != block2.length() {
    raise InvalidReferenceShape
  }
  ReadOnlyArray::from_array(
    Array::makei(block0.length(), index => {
      let sum01 = block0[index] + block1[index]
      let sum012 = sum01 + block2[index]
      sum012 / 3.0F
    }),
  )
}

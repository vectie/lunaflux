///|
fn reusable_runtime_validate_head_ownership(
  abi : ReusableFusedRuntimeKernelAbi,
  heads : Int,
  rows : Int?,
) -> Unit raise FusedArtifactAdmissionError {
  if !(heads is (1 | 2 | 4)) ||
    (
      heads != 1 &&
      (
        abi != QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 ||
        rows is None
      )
    ) {
    raise InvalidFusedArtifact(Launch)
  }
}

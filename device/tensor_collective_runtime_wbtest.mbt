///|
test "collective runtime inventory maps every private availability state" {
  let cases = [
    (@nccl.Available, CollectiveRuntimeAvailable),
    (UnsupportedPlatform, CollectiveRuntimeUnsupportedPlatform),
    (LibraryMissing, CollectiveRuntimeLibraryMissing),
    (AbiIncomplete, CollectiveRuntimeAbiIncomplete),
    (VersionUnavailable, CollectiveRuntimeVersionUnavailable),
  ]
  for test_case in cases {
    let (internal, expected) = test_case
    assert_eq(collective_availability_from_internal(internal), expected)
  }
}

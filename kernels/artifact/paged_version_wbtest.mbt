///|
test "paged artifact admission fails closed outside catalog v3" {
  try ensure_paged_catalog_version(@catalog.CatalogVersion::v1()) catch {
    CatalogVersionMismatch(expected~, actual~) => {
      assert_eq(expected, @catalog.CatalogVersion::v3())
      assert_eq(actual, @catalog.CatalogVersion::v1())
    }
    _ => fail("unexpected catalog-version error")
  } noraise {
    _ => fail("expected catalog-version rejection")
  }
}

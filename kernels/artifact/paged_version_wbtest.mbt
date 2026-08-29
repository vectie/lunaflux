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

///|
test "stateless artifact admission fails closed outside catalog v1" {
  try ensure_stateless_catalog_version(@catalog.CatalogVersion::v4()) catch {
    CatalogVersionMismatch(expected~, actual~) => {
      assert_eq(expected, @catalog.CatalogVersion::v1())
      assert_eq(actual, @catalog.CatalogVersion::v4())
    }
    _ => fail("unexpected stateless catalog-version error")
  } noraise {
    _ => fail("legacy artifact path accepted catalog v4")
  }
}

///|
test "catalog-v4 artifact admission fails closed outside catalog v4" {
  try ensure_paged_v4_catalog_version(@catalog.CatalogVersion::v3()) catch {
    CatalogVersionMismatch(expected~, actual~) => {
      assert_eq(expected, @catalog.CatalogVersion::v4())
      assert_eq(actual, @catalog.CatalogVersion::v3())
    }
    _ => fail("unexpected catalog-v4 version error")
  } noraise {
    _ => fail("catalog-v4 artifact path accepted catalog v3")
  }
}

///|
test "catalog-v4 artifact admission bounds total comparison work" {
  try ensure_paged_v4_artifact_contract_count(1025) catch {
    ContractCountExceeded(maximum~) => assert_eq(maximum, 1024)
    _ => fail("unexpected catalog-v4 work-bound error")
  } noraise {
    _ => fail("catalog-v4 artifact path accepted hostile contract cardinality")
  }
  ensure_paged_v4_artifact_contract_count(1024)
}

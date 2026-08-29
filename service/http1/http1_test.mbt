///|
test "public Luna HTTP/1 surface exposes only route and indexed body" {
  let expected : FixedArray[Byte] = [b'k']
  let policy = try! @api_auth.LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=1,
    maximum_credential_bytes=1,
  )
  let limits = try! LunaHttp1Limits::new(max_head_bytes=256, max_body_bytes=32)
  assert_eq(try! LunaHttp1Workspace::required_byte_cells(limits), 288UL)
  assert_eq(try! LunaHttp1Workspace::required_reference_cells(limits), 2UL)
  let workspace = try! LunaHttp1Workspace::new(
    limits,
    try! LunaHttp1StepBudget::new(1),
    policy,
  )
  let work = try! workspace.begin()
  assert_eq(try! work.state(), LunaHttp1ReceivingHead)
  try! work.abort()
}

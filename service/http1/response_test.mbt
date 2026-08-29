///|
test "public Luna HTTP/1 response surface is bounded and opaque" {
  let workspace = LunaHttp1ResponseWorkspace::new(
    try! LunaHttp1ResponseStepBudget::new(17),
  )
  assert_eq(LunaHttp1ResponseWorkspace::required_byte_cells(), 167UL)
  let work = try! workspace.begin_error(LunaHttp1ValidationError)
  assert_eq(try! work.state(), LunaHttp1ResponseBuilding)
  try! work.abort()
}

///|
test "partial function acquisition retains all acquired handles" {
  let handles : Array[String] = []
  try
    acquire_functions(["q", "k", "v"], handles, name => {
      if name == "v" {
        raise @device.DeviceError::InvalidArgument(FunctionLoad)
      }
      name
    })
  catch {
    @device.DeviceError::InvalidArgument(FunctionLoad) => ()
    _ => fail("unexpected load error")
  } noraise {
    _ => fail("expected missing symbol")
  }
  assert_eq(handles.length(), 2)
  let released : Array[String] = []
  release_functions(handles, name => released.push(name))
  assert_eq(released[0], "k")
  assert_eq(released[1], "q")
  assert_true(handles.is_empty())
  release_functions(handles, _ => {
    raise @device.DeviceError::InvalidArgument(FunctionClose)
  })
}

///|
test "failed release retains failing and earlier handles for retry" {
  let handles = ["q", "k", "v"]
  let released : Array[String] = []
  try
    release_functions(handles, name => {
      if name == "k" {
        raise @device.DeviceError::InvalidArgument(FunctionClose)
      }
      released.push(name)
    })
  catch {
    @device.DeviceError::InvalidArgument(FunctionClose) => ()
    _ => fail("unexpected close error")
  } noraise {
    _ => fail("expected busy function")
  }
  assert_eq(handles.length(), 2)
  assert_eq(released[0], "v")
  release_functions(handles, name => released.push(name))
  assert_eq(released.length(), 3)
  assert_eq(released[1], "k")
  assert_eq(released[2], "q")
}

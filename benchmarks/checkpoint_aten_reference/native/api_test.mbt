///|
test "missing independent backend library raises and releases the loader" {
  try
    @native.Session::new(
      "/lunaflux-missing-test-only-aten-library.so",
      device=0,
      budget=1024L,
    )
  catch {
    _ => ()
  } noraise {
    session => {
      session.close()
      fail("missing backend must not produce an initialized session")
    }
  }
}

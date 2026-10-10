///|
test "variant component selection has no scheduler or hardware dependence" {
  let (kind, path) = component("transformer", Fl2va)
  assert_true(kind == Fl2vaDenoiser)
  assert_eq(path, "transformer")
  assert_true(component("transformer", Ref2va).0 == Ref2vaDenoiser)
  assert_eq(component("video_vae", Fl2va).1, "video_vae/source")
  try component("unknown", Fl2va) catch {
    _ => ()
  } noraise {
    _ => fail("unknown checkpoint component accepted")
  }
}

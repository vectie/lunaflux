///|
test "device worker rejects unsafe model locators during inert admission" {
  for
    locator in [
      "", "/weights.safetensors", "../weights.safetensors", "models/../weights.safetensors",
      "models//weights.safetensors", "models/./weights.safetensors", "models\\weights.safetensors",
    ] {
    try admit_model_locator(locator) catch {
      Invalid(ModelLocator) => ()
      _ => fail("unexpected model-locator admission error")
    } noraise {
      _ => fail("unsafe model locator was admitted")
    }
  }
}

///|
test "device worker admits canonical model locator without filesystem access" {
  try admit_model_locator("models/weights.safetensors") catch {
    _ => fail("canonical model locator was rejected")
  } noraise {
    _ => ()
  }
}

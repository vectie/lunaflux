///|
test "complete component packings are bounded and retain all model blocks" {
  let metadata = @h3.MiniMaxH3ModelMetadata::from_verified_content(
    @h3.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    @spec.ContentDigest::from_sha256(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
  )
  let plan = @minimax_h3.build(metadata)
  let ceiling = 103079215104L
  let components : Array[@weights.MiniMaxH3WeightComponent] = [
    Fl2vaDenoiser,
    TextConditioner,
    VideoVae,
    AudioVae,
  ]
  for component in components {
    let packings = @minimax_h3_checkpoint.component_packings(
      component,
      plan,
      maximum_device_bytes=ceiling,
    )
    let n = @startup.component_bytes(component, packings[:], ceiling)
    assert_true(n > 0L && n < ceiling)
    assert_eq(
      @startup.component_bytes(
        component,
        @minimax_h3_checkpoint.component_packings(
          component,
          plan,
          maximum_device_bytes=n,
        )[:],
        n,
      ),
      n,
    )
    if component == Fl2vaDenoiser {
      assert_eq(packings.length(), 56)
      assert_eq(packings.filter(p => p is Block(_)).length(), 50)
    }
    if component == TextConditioner {
      assert_eq(packings.length(), 2)
    }
    try
      @minimax_h3_checkpoint.component_packings(
        component,
        plan,
        maximum_device_bytes=n - 1L,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("component admitted beyond its actual packed footprint")
    }
  }
}

///|
test "available CUDA device receives direct aligned tensor copies" {
  match @device.probe() {
    Unavailable(_) => ()
    Available(devices~, ..) => {
      let context = @device.open_context(devices[0].ordinal())
      let bindings = test_bindings()
      let file = test_file(bindings.materialized_bytes().to_int())
      let weights = @device_materialize.load(
        context,
        bindings,
        file,
        materialization_limits=test_limits(),
        alignment_bytes=64L,
        max_arena_bytes=4096L,
      )
      assert_eq(weights.identity(), bindings.identity())
      assert_eq(weights.layout().arena_bytes(), 768L)
      assert_eq(weights.allocation().byte_count(), 768L)
      let first = try! @plan.TensorRef::new(0)
      guard weights.region(first) is Some(region) else {
        fail("expected first region")
      }
      assert_eq(
        weights
        .allocation()
        .copy_to_host(region.offset_bytes(), region.byte_length()),
        file[24:88].to_owned(),
      )
      weights.close()
      context.close()
    }
  }
}

///|
test "malformed input is rejected after context admission but before loading" {
  match @device.probe() {
    Unavailable(_) => ()
    Available(devices~, ..) => {
      let context = @device.open_context(devices[0].ordinal())
      try
        @device_materialize.load(
          context,
          test_bindings(),
          b"short",
          materialization_limits=test_limits(),
          alignment_bytes=64L,
          max_arena_bytes=4096L,
        )
      catch {
        @device_materialize.DeviceWeightError::MaterializationFailed(
          MissingHeaderLength
        ) => ()
        _ => fail("unexpected malformed-input error")
      } noraise {
        _ => fail("expected malformed-input rejection")
      }
      context.close()
    }
  }
}

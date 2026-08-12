///|
test "probe is deterministic and semantically bounded on every host" {
  let first = @device.probe()
  assert_eq(@device.probe(), first)
  match first {
    Unavailable(reason) =>
      match reason {
        UnsupportedPlatform
        | DriverLibraryMissing
        | DriverAbiIncomplete
        | DriverInitializationFailed
        | NoDevices
        | InventoryFailed => ()
      }
    Available(driver_version~, devices~) => {
      assert_true(driver_version > 0)
      assert_true(devices.length() > 0)
      for index, info in devices {
        assert_eq(info.ordinal(), index)
        assert_true(info.name().length() > 0)
        assert_true(info.capability().total_memory_bytes() > 0L)
        assert_true(info.capability().compute_major() > 0)
        assert_true(info.capability().compute_minor() >= 0)
      }
    }
  }
}

///|
test "context admission exposes only bounded public failures" {
  match @device.probe() {
    Unavailable(_) =>
      try @device.open_context(0) catch {
        Unavailable(ContextCreate) => ()
        error => fail("unexpected public context error: \{Repr(error)}")
      } noraise {
        _ => fail("unavailable device unexpectedly opened a context")
      }
    Available(_) =>
      try @device.open_context(-1) catch {
        InvalidArgument(ContextCreate) => ()
        error => fail("unexpected public context error: \{Repr(error)}")
      } noraise {
        _ => fail("negative device ordinal unexpectedly opened a context")
      }
  }
}

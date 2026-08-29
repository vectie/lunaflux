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
        assert_true(info.uuid().length() > 0)
        assert_true(info.pci().length() > 0)
        assert_true(info.capability().total_memory_bytes() > 0L)
        assert_true(info.capability().compute_major() > 0)
        assert_true(info.capability().compute_minor() >= 0)
      }
    }
  }
}

///|
test "opened context retains the exact probed ordinal identity without authority" {
  match @device.probe() {
    Unavailable(_) => ()
    Available(devices~, ..) => {
      let expected = devices[0]
      let context = @device.open_context(expected.ordinal())
      let identity = context.device_identity()
      assert_eq(identity.ordinal(), expected.ordinal())
      assert_eq(identity.name(), expected.name())
      assert_eq(identity.uuid(), expected.uuid())
      assert_eq(identity.pci(), expected.pci())
      assert_eq(identity.capability(), expected.capability())
      assert_eq(
        identity.total_memory_bytes(),
        expected.capability().total_memory_bytes(),
      )
      context.close()
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

///|
test "directed peer inventory preserves bounded host availability" {
  match @device.probe() {
    Unavailable(_) =>
      try @device.can_access_peer(0, 1) catch {
        Unavailable(DevicePeerAccess) => ()
        error => fail("unexpected unavailable peer error: \{Repr(error)}")
      } noraise {
        _ => fail("unavailable host published peer access")
      }
    Available(_) =>
      try @device.can_access_peer(0, 0) catch {
        InvalidArgument(DevicePeerAccess) => ()
        error => fail("unexpected invalid peer error: \{Repr(error)}")
      } noraise {
        _ => fail("self peer query bypassed ordinal admission")
      }
  }
}

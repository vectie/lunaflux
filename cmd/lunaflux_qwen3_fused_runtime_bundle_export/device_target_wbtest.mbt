///|
test "recipe device target survives fused and unfused ingress selection" {
  for choice in ["full", "unfused"] {
    let raw = runtime_bundle_test_arguments("5")
    raw.append(["--ingress-evaluate", choice, "--device-target", "12", "1"])
    let (args, _) = prepare_ingress_choice(raw)
    assert_eq(
      runtime_device_target(args),
      @catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=1,
        supports_bf16=true,
        supports_cublas_lt=false,
      ),
    )
    raw.append(["--device-target", "12", "0"])
    try prepare_ingress_choice(raw) catch {
      InvalidArguments => ()
      _ => fail("unexpected device target error")
    } noraise {
      _ => fail("duplicate target accepted")
    }
  }
}

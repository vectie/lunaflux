///|
fn print_usage() -> Unit {
  println("LunaFlux \{@contracts.protocol_version()}")
  println("usage:")
  println("  lunaflux doctor")
  println("  lunaflux plan MODEL_PATH sha256:DIGEST")
  println("  lunaflux version")
}

///|
fn unavailable_device_reason(reason : @hardware.UnavailableReason) -> String {
  match reason {
    UnsupportedPlatform => "unsupported platform"
    DriverLibraryMissing => "driver library missing"
    DriverAbiIncomplete => "driver ABI incomplete"
    DriverInitializationFailed => "driver initialization failed"
    NoDevices => "no devices"
    InventoryFailed => "inventory failed"
  }
}

///|
fn print_doctor() -> Unit {
  println("LunaFlux foundation inspection")
  println("contracts: ready")
  println("configuration validators: ready")
  println("bounded tokenizer JSON admission: ready")
  println("bounded model metadata and weight-manifest validation: ready")
  match @hardware.probe() {
    Available(driver_version~, devices~) =>
      println(
        "CUDA device probe: available; driver \{driver_version}; devices \{devices.length()}",
      )
    Unavailable(reason) =>
      println(
        "CUDA device probe: unavailable; \{unavailable_device_reason(reason)}",
      )
  }
  println("offline reference executor: ready (correctness only)")
  println("model file loading: not implemented")
  println("production device execution: not implemented")
  println("readiness: false")
}

///|
fn print_plan(model_path : String, digest : String) -> Unit raise {
  let service = @service.new("127.0.0.1", 8080, 1048576, 30000, Normal)
  let model = @model.new(model_path, digest, ["LlamaForCausalLM"])
  let device = @device.new([0], 8589934592L)
  let plan = @resolved.resolve(service, model, device)
  println("LunaFlux configuration-only startup plan")
  println("artifact identity: validated; path redacted")
  println("architecture: \{plan.selected_architecture()}")
  println("primary device: \{plan.primary_device()} (configured, not probed)")
  println("memory ceiling bytes: \{plan.memory_ceiling_bytes()}")
  println("decisions recorded: \{plan.decisions().length()}")
  println("model materialization: not performed")
  println("readiness: false; CUDA capability resolution is a later gate")
}

///|
fn main raise {
  match @env.args() {
    [_, "doctor"] => print_doctor()
    [_, "plan", model_path, digest] => print_plan(model_path, digest)
    [_, "version"] => println(@contracts.protocol_version())
    _ => print_usage()
  }
}

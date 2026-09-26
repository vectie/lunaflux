///|
priv struct StreamingClaims {
  slots : Int
  host_bytes : Int64
  reserve_pages : Int
  restore_ns : Int64
  recompute_ns : Int64
  separate_host_memory : Bool
  transfer_timeout_millis : Int
  workspace_budget : Int64
  shapes : ReadOnlyArray[@streaming_ir.Shape]
}

///|
fn parse_streaming(
  root : Map[String, Json],
) -> StreamingClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "streaming", Execution)
  validate_fields(object, Execution, [
    "host_slots", "host_budget_bytes", "reserve_pages", "restore_nanoseconds_per_page",
    "recompute_nanoseconds_per_page", "separate_host_memory", "transfer_timeout_millis",
    "workspace_budget_bytes", "prefill_shapes",
  ])
  guard object.get("prefill_shapes") is Some(Array(values)) else {
    raise invalid_field(Execution, ExpectedObject)
  }
  if values.length() < 1 || values.length() > 32 {
    raise invalid_field(Execution, UnsupportedValue)
  }
  let shapes : Array[@streaming_ir.Shape] = []
  for value in values {
    guard value is Object(shape) else {
      raise invalid_field(Execution, ExpectedObject)
    }
    validate_fields(shape, Execution, ["tokens", "workspace_bytes"])
    shapes.push(
      @streaming_ir.Shape::new(
        tokens=require_int(shape, "tokens", Execution),
        workspace_bytes=require_int64(shape, "workspace_bytes", Execution),
      ) catch {
        _ => raise invalid_field(Execution, UnsupportedValue)
      },
    )
  }
  {
    transfer_timeout_millis: require_int(
      object,
      "transfer_timeout_millis",
      Execution,
    ),
    workspace_budget: require_int64(object, "workspace_budget_bytes", Execution),
    shapes: ReadOnlyArray::from_array(shapes),
    slots: require_int(object, "host_slots", Execution),
    host_bytes: require_int64(object, "host_budget_bytes", Execution),
    reserve_pages: require_int(object, "reserve_pages", Execution),
    restore_ns: require_int64(object, "restore_nanoseconds_per_page", Execution),
    recompute_ns: require_int64(
      object,
      "recompute_nanoseconds_per_page",
      Execution,
    ),
    separate_host_memory: require_bool(
      object,
      "separate_host_memory",
      Execution,
    ),
  }
}

///|
fn admit_streaming(
  claims : StreamingClaims,
  layout : @device_layout.DeviceKvLayout,
) -> @streaming_ir.Capability raise RuntimeDescriptorFileError {
  let components = layout.geometry().layer_count().to_int64() * 2L
  if components <= 0L ||
    layout.page_logical_bytes() > 9223372036854775807L / components {
    raise AdmissionRejected(KvLayout)
  }
  @streaming_ir.Capability::new(
    slots=claims.slots,
    host_bytes=claims.host_bytes,
    page_bytes=layout.page_logical_bytes() * components,
    page_count=layout.total_page_count(),
    reserve_pages=claims.reserve_pages,
    restore_ns=claims.restore_ns,
    recompute_ns=claims.recompute_ns,
    separate_host_memory=claims.separate_host_memory,
    transfer_timeout_millis=claims.transfer_timeout_millis,
    workspace_budget=claims.workspace_budget,
    shapes=claims.shapes[:],
  ) catch {
    _ => raise AdmissionRejected(StartupContract)
  }
}

///|
pub(all) suberror TwoSparkWorkerError {
  Configuration
  Digest
  Admission
  Cleanup
}

///|
fn text(object : Map[String, Json], name : String) -> String raise {
  guard object.get(name) is Some(String(value)) else { raise Configuration }
  value
}

///|
fn number(object : Map[String, Json], name : String) -> Int64 raise {
  let value = @string.parse_int64(text(object, name))
  if value < 0L {
    raise Configuration
  }
  value
}

///|
fn fields(object : Map[String, Json], names : ArrayView[String]) -> Unit raise {
  if object.length() != names.length() {
    raise Configuration
  }
  for name in names {
    if !object.contains(name) {
      raise Configuration
    }
  }
}

///|
fn snapshot(
  root : @approved_fs.ApprovedRoot,
  locator : String,
  maximum : Int64,
) -> Bytes raise {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(locator))
  defer (file.close() catch { _ => () })
  file.read_immutable_snapshot(maximum_bytes=maximum)
}

///|
fn fixed(bytes : Bytes) -> FixedArray[Byte] {
  FixedArray::makei(bytes.length(), i => bytes[i])
}

///|
fn pinned(
  root : @approved_fs.ApprovedRoot,
  reference : Json,
  maximum : Int64,
) -> Bytes raise {
  guard reference is Object(object) else { raise Configuration }
  fields(object, ["path", "sha256"])
  let bytes = snapshot(root, text(object, "path"), maximum)
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) !=
    text(object, "sha256") {
    raise Digest
  }
  bytes
}

///|
fn field(object : Map[String, Json], name : String) -> Json raise {
  guard object.get(name) is Some(value) else { raise Configuration }
  value
}

///|
fn pair(object : Map[String, Json], name : String) -> Array[Json] raise {
  guard field(object, name) is Array(values) else { raise Configuration }
  if values.length() != 2 {
    raise Configuration
  }
  values
}

///|
fn worker_contract(
  root : @approved_fs.ApprovedRoot,
  reference : Json,
) -> @worker_wire.WorkerStartupContract raise {
  let bytes = pinned(root, reference, 408L)
  let frame = @worker_wire.StartupFrameBuffer::new().load(
    fixed(bytes),
    bytes.length(),
  )
  if frame.kind() != Configure {
    raise Configuration
  }
  frame.contract()
}

///|
fn network_node(
  value : Json,
  target : @catalog.DeviceTarget,
) -> @network_topology.NetworkRank raise {
  guard value is Object(object) else { raise Configuration }
  fields(object, [
    "node", "device_ordinal", "physical_memory_bytes", "rdma_interface",
  ])
  let ordinal = number(object, "device_ordinal")
  if ordinal > 1023L {
    raise Configuration
  }
  @network_topology.NetworkRank::new(
    node=@network_topology.NodeIdentity::new(text(object, "node")),
    device_ordinal=ordinal.to_int(),
    target~,
    physical_memory_bytes=number(object, "physical_memory_bytes"),
    rdma_interface=text(object, "rdma_interface"),
  )
}

///|
fn memory(value : Json) -> @two_node_plan.RankMemoryBudget raise {
  guard value is Object(object) else { raise Configuration }
  fields(object, [
    "system_reserved_bytes", "runtime_ceiling_bytes", "activation_workspace_bytes",
    "collective_bytes", "kv_bytes",
  ])
  @two_node_plan.RankMemoryBudget::new(
    system_reserved_bytes=number(object, "system_reserved_bytes"),
    runtime_ceiling_bytes=number(object, "runtime_ceiling_bytes"),
    activation_workspace_bytes=number(object, "activation_workspace_bytes"),
    collective_bytes=number(object, "collective_bytes"),
    kv_bytes=number(object, "kv_bytes"),
  )
}

///|
fn document(bytes : Bytes, expected_sha256 : String) -> Map[String, Json] raise {
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != expected_sha256 {
    raise Digest
  }
  let decoded = @utf8.decode(bytes)
  @json_guard.reject_duplicate_object_keys(decoded, max_depth=8)
  guard @json.parse(decoded) is Object(object) else { raise Configuration }
  fields(object, [
    "schema", "source", "workers", "envelope", "nodes", "budgets", "executables",
    "deployment_sha256", "lease_millis", "rank", "model_root", "kernel_root", "listen",
    "certificate", "private_key", "credential", "max_payload_bytes",
  ])
  if text(object, "schema") != "lunaflux.two-spark-worker.v1" {
    raise Configuration
  }
  object
}

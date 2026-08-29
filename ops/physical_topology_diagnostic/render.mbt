///|
fn append_field(output : StringBuilder, label : String, value : String) -> Unit {
  output.write_string(label)
  output.write_string(": ")
  output.write_string(value)
  output.write_string("\n")
}

///|
fn bool_name(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn hex_nibble(value : Int) -> Char {
  match value {
    0 => '0'
    1 => '1'
    2 => '2'
    3 => '3'
    4 => '4'
    5 => '5'
    6 => '6'
    7 => '7'
    8 => '8'
    9 => '9'
    10 => 'a'
    11 => 'b'
    12 => 'c'
    13 => 'd'
    14 => 'e'
    _ => 'f'
  }
}

///|
fn utf8_hex(value : StringView) -> String {
  let output = StringBuilder()
  for unit in @utf8.encode(value) {
    output.write_char(hex_nibble(unit.to_int() / 16))
    output.write_char(hex_nibble(unit.to_int() % 16))
  }
  output.to_string()
}

///|
fn collective_availability_name(
  value : @device.TensorCollectiveRuntimeAvailability,
) -> String {
  match value {
    CollectiveRuntimeAvailable => "available"
    CollectiveRuntimeUnsupportedPlatform => "unsupported_platform"
    CollectiveRuntimeLibraryMissing => "library_missing"
    CollectiveRuntimeAbiIncomplete => "abi_incomplete"
    CollectiveRuntimeVersionUnavailable => "version_unavailable"
  }
}

///|
fn unavailable_name(value : @device.UnavailableReason) -> String {
  match value {
    UnsupportedPlatform => "unsupported_platform"
    DriverLibraryMissing => "driver_library_missing"
    DriverAbiIncomplete => "driver_abi_incomplete"
    DriverInitializationFailed => "driver_initialization_failed"
    NoDevices => "no_devices"
    InventoryFailed => "inventory_failed"
  }
}

///|
fn stage_name(value : PhysicalTopologyDiagnosticStage) -> String {
  match value {
    DeviceProbe => "device_probe"
    PeerProbe => "peer_probe"
    TopologyProjection => "topology_projection"
    TopologyDeclaration => "topology_declaration"
    TopologyAdmission => "topology_admission"
    CollectiveRuntime => "collective_runtime"
  }
}

///|
fn area_name(value : @device_topology.LocalTopologyArea) -> String {
  match value {
    DeclarationRanks => "declaration_ranks"
    DeclarationLinks => "declaration_links"
    ProbeDevices => "probe_devices"
    ProbeLinks => "probe_links"
    AdmissionRanks => "admission_ranks"
    AdmissionLinks => "admission_links"
  }
}

///|
fn rule_name(value : @device_topology.LocalTopologyRule) -> String {
  match value {
    TooFew => "too_few"
    TooMany => "too_many"
    InvalidOrdinal => "invalid_ordinal"
    InvalidName => "invalid_name"
    InvalidMemory => "invalid_memory"
    RankOrder => "rank_order"
    Duplicate => "duplicate"
    Incomplete => "incomplete"
    Unsupported => "unsupported"
    HeterogeneousTarget => "heterogeneous_target"
    ProbeUnavailable => "probe_unavailable"
    CountMismatch => "count_mismatch"
    IdentityMismatch => "identity_mismatch"
    TargetMismatch => "target_mismatch"
  }
}

///|
fn append_rejection(
  output : StringBuilder,
  rejection : PhysicalTopologyDiagnosticRejection,
) -> Unit {
  match rejection {
    DeviceUnavailable(reason) => {
      append_field(output, "rejection.stage", "device_probe")
      append_field(output, "rejection.reason", unavailable_name(reason))
    }
    PeerQueryFailed(source_rank~, destination_rank~) => {
      append_field(output, "rejection.stage", "peer_probe")
      append_field(output, "rejection.reason", "query_failed")
      append_field(output, "rejection.first", source_rank.to_string())
      append_field(output, "rejection.second", destination_rank.to_string())
    }
    TopologyRejected(stage~, area~, rule~, first~, second~) => {
      append_field(output, "rejection.stage", stage_name(stage))
      append_field(output, "rejection.area", area_name(area))
      append_field(output, "rejection.reason", rule_name(rule))
      append_field(output, "rejection.first", first.to_string())
      append_field(output, "rejection.second", second.to_string())
    }
    CollectiveUnavailable(availability) => {
      append_field(output, "rejection.stage", "collective_runtime")
      append_field(
        output,
        "rejection.reason",
        collective_availability_name(availability),
      )
    }
    CollectiveVersionRejected => {
      append_field(output, "rejection.stage", "collective_runtime")
      append_field(output, "rejection.reason", "version_rejected")
    }
  }
}

///|
/// Renders the immutable schema in fixed field and canonical matrix order.
/// Device names are exact UTF-8 hex so control characters cannot forge fields.
pub fn render(evidence : PhysicalTopologyDiagnosticEvidence) -> String {
  let output = StringBuilder()
  append_field(output, "schema", "lunaflux.phase7.physical_topology.v1")
  append_field(
    output,
    "outcome",
    if evidence.status() == PhysicalTopologySupported {
      "supported"
    } else {
      "rejected"
    },
  )
  append_field(output, "driver.version", evidence.driver_version.to_string())
  append_field(output, "device.count", evidence.devices.length().to_string())
  for index, device in evidence.devices {
    let prefix = "device.\{index}."
    let capability = device.capability
    append_field(output, prefix + "ordinal", device.ordinal.to_string())
    append_field(output, prefix + "name_utf8_hex", utf8_hex(device.name))
    append_field(
      output,
      prefix + "compute_major",
      capability.compute_major().to_string(),
    )
    append_field(
      output,
      prefix + "compute_minor",
      capability.compute_minor().to_string(),
    )
    append_field(
      output,
      prefix + "total_memory_bytes",
      capability.total_memory_bytes().to_string(),
    )
    append_field(
      output,
      prefix + "supports_bf16",
      bool_name(capability.supports_bf16()),
    )
    append_field(
      output,
      prefix + "supports_cublas_lt",
      bool_name(capability.supports_cublas_lt()),
    )
  }
  append_field(
    output,
    "peer.query_complete",
    bool_name(evidence.peer_query_complete),
  )
  append_field(output, "peer.count", evidence.peers.length().to_string())
  for index, peer in evidence.peers {
    let prefix = "peer.\{index}."
    append_field(
      output,
      prefix + "source_ordinal",
      peer.source_ordinal.to_string(),
    )
    append_field(
      output,
      prefix + "destination_ordinal",
      peer.destination_ordinal.to_string(),
    )
    append_field(output, prefix + "accessible", bool_name(peer.accessible))
  }
  append_field(
    output,
    "collective.required_minimum_version",
    MINIMUM_COLLECTIVE_RUNTIME_VERSION.to_string(),
  )
  append_field(
    output,
    "collective.required_maximum_version",
    MAXIMUM_COLLECTIVE_RUNTIME_VERSION.to_string(),
  )
  append_field(
    output,
    "collective.availability",
    collective_availability_name(evidence.collective.availability),
  )
  append_field(
    output,
    "collective.observed_version",
    evidence.collective.version_code.to_string(),
  )
  append_field(
    output,
    "collective.admitted_version",
    evidence.collective.admitted_version_code.to_string(),
  )
  append_field(
    output,
    "topology.admitted",
    bool_name(evidence.admitted_topology is Some(_)),
  )
  match evidence.rejection {
    Some(rejection) => append_rejection(output, rejection)
    None => {
      append_field(output, "rejection.stage", "none")
      append_field(output, "rejection.reason", "none")
    }
  }
  append_field(output, "authority.context_opened", "false")
  append_field(output, "authority.device_allocation_created", "false")
  append_field(output, "authority.communicator_created", "false")
  append_field(output, "authority.rank_process_spawned", "false")
  output.to_string()
}

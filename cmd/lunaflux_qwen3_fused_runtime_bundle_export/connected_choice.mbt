///|
/// Offline locator association, not a checksum of the referenced CUBIN.
/// Replacing code at a locator requires a fresh locator or externally supplied
/// tuning scope. No file bytes are read or authenticated to compute this label.
fn runtime_module_locator_label(path : String) -> String {
  @crypto.bytes_to_hex_string(
    @crypto.sha256(@utf8.encode("fused-module-locator-v1\n" + path)),
  )
}

///|
priv struct PrefillChainMeasurements {
  context_sha256 : String
  modules : ReadOnlyArray[(String, String)]
  route_files : ReadOnlyArray[(String, String)?]
  costs : ReadOnlyArray[@fusion_regions.Cost]
  vendor_rows : Int
}

///|
/// Export-only comparison. The hash commits the fixed command context, with
/// only executable name and destination removed. Measurement files additionally
/// carry their own artifact/workload/device provenance, sealed by the caller.
fn chain_context_sha256(args : ArrayView[String], output : String) -> String {
  let canonical = args[1:].map(value => {
    if value == output {
      "<output>"
    } else {
      value
    }
  })
  @crypto.bytes_to_hex_string(
    @crypto.sha256(@utf8.encode(Json(canonical).stringify())),
  )
}

///|
fn parse_prefill_chain_measurements(
  bytes : Bytes,
) -> PrefillChainMeasurements raise Qwen3FusedRuntimeBundleExportError {
  let value = @json.parse(@utf8.decode(bytes)) catch {
    _ => raise InvalidArguments
  }
  guard value
    is {
      "schema": "lunaflux-prefill-connected-comparison.v1",
      "context_sha256": String(context),
      "numeric_law": "reference-bf16-probability-exp2-v1",
      "measurement_scope": String(scope),
      "vendor_rows": String(rows),
      "modules": Array(modules),
      "observations": Array(observations),
      ..
    } &&
    lower_sha256(context) &&
    !scope.is_empty() &&
    modules.length() == 2 &&
    observations.length() == 4 else {
    raise InvalidArguments
  }
  let route_files = modules.map(entry => {
    match entry {
      { "routes_path": String(path), "routes_sha256": String(digest), .. } => {
        if !safe_absolute_file(path) || !lower_sha256(digest) {
          raise InvalidArguments
        }
        Some((path, digest))
      }
      { "routes_path": _, .. } | { "routes_sha256": _, .. } =>
        raise InvalidArguments
      _ => None
    }
  })
  let modules = modules.map(entry => {
    guard entry is { "path": String(path), "sha256": String(digest), .. } &&
      safe_absolute_file(path) &&
      lower_sha256(digest) else {
      raise InvalidArguments
    }
    (path, digest)
  })
  if modules[0].0 == modules[1].0 || modules[0].1 == modules[1].1 {
    raise InvalidArguments
  }
  let costs = observations.mapi((index, observation) => {
    guard observation
      is {
        "plan": String(id),
        "nanoseconds": String(ns),
        "samples": String(samples),
        ..
      } &&
      id == index.to_string() else {
      raise InvalidArguments
    }
    @fusion_regions.Cost::measured(
      nanoseconds=positive_nanoseconds(ns),
      samples=positive_int(samples),
    ) catch {
      _ => raise InvalidArguments
    }
  })
  let vendor_rows = positive_int(rows)
  if vendor_rows < 2 || vendor_rows > 1048576 {
    raise InvalidArguments
  }
  {
    context_sha256: context,
    modules: ReadOnlyArray::from_array(modules),
    route_files: ReadOnlyArray::from_array(route_files),
    costs: ReadOnlyArray::from_array(costs),
    vendor_rows,
  }
}

///|
async fn read_connected_choice(
  raw : ArrayView[String],
) -> (ArrayView[String], PrefillChainMeasurements?) raise Qwen3FusedRuntimeBundleExportError {
  guard raw is [.. rest, "--prefill-chain-comparison", path, digest] else {
    if raw.contains("--prefill-chain-comparison") {
      raise InvalidArguments
    }
    return (raw, None)
  }
  if rest.contains("--prefill-chain-comparison") ||
    rest.contains("--prefill-vendor-projection-rows") ||
    rest.contains("--attention-buckets") ||
    !safe_absolute_file(path) ||
    !lower_sha256(digest) {
    raise InvalidArguments
  }
  let bytes = read_module(path)
  if bytes.length() > 65536 {
    raise InvalidModule
  }
  let measured = parse_prefill_chain_measurements(bytes)
  // Read selected modules once when building the bundle. Do not open and hash
  // every competing module merely to authenticate supplied measurement labels.
  (rest, Some(measured))
}

///|
/// Routes belong to the chosen alternative. Normal bundle admission still
/// checks their exact source/launch scope; this does not relabel measurements.
async fn connected_route_bytes(
  measured : PrefillChainMeasurements,
  plan : Int,
) -> Bytes? raise Qwen3FusedRuntimeBundleExportError {
  guard measured.route_files[plan / 2] is Some((path, _)) else { return None }
  let bytes = read_module(path)
  if bytes.length() > 1048576 {
    raise InvalidModule
  }
  Some(bytes)
}

///|
/// Both implementations retain the same BF16 materialized live boundary. The
/// consumer spans the remaining execution, so observations must time that
/// complete continuation (not add separately measured kernels).
fn bind_connected_choice(
  raw : ArrayView[String],
  args : RuntimeBundleArguments,
  ingress : @luna_fusion_strategy.FusionSchedule,
  law : @luna_fused_artifact_admission.PrefillExponentialLaw,
  measured : PrefillChainMeasurements,
) -> (RuntimeBundleArguments, Int, Int) raise Qwen3FusedRuntimeBundleExportError {
  if law != ReferenceBf16ProbabilityBaseTwo ||
    chain_context_sha256(raw, args.output_path) != measured.context_sha256 {
    raise InvalidArguments
  }
  validate_reference_prefill_choice(args, ingress)
  let selected = try {
    let graph = @fusion_regions.Graph::new([
      @fusion_regions.Node::new([], observable=false, effect=Pure),
      @fusion_regions.Node::new([0], observable=true, effect=Ordered),
    ])
    let plans = [
      for id in 0..<4 => {
        let producer = id / 2
        let consumer = id % 2
        @fusion_regions.ConnectedPlan::new(id~, graph, [
          @fusion_regions.RegionImplementation::new(
            id=producer,
            @fusion_regions.Region::new(graph, start=0, count=1, [0]),
            input=Materialized,
            output=Scoped(0),
            workspace_bytes=0L,
            local_bytes=Some(if producer == 0 { 81920L } else { 49152L }),
          ),
          @fusion_regions.RegionImplementation::new(
            id=consumer + 2,
            @fusion_regions.Region::new(graph, start=1, count=1, [1]),
            input=Scoped(0),
            output=Materialized,
            // Current ordered-projection ABI owns a 32 MiB workspace. Persistent
            // tensors and startup warmup storage remain in the runtime plan.
            workspace_bytes=if consumer == 1 { 33554432L } else { 0L },
            // Vendor/private projection residency is opaque to this pass.
            // Unknown is not zero; existing artifact/runtime admission owns it.
            local_bytes=None,
          ),
        ])
      }
    ]
    @fusion_regions.select_connected_plan(
      plans,
      measured.costs[:],
      maximum_workspace_bytes=33554432L,
      maximum_local_bytes=None,
    ).implementation_ids()
  } catch {
    _ => raise InvalidArguments
  }
  let attention = selected[0]
  let vendor = selected[1] == 3
  let shared = if attention == 0 { 81920 } else { 49152 }
  let path = measured.modules[attention].0
  let args = {
    ..args,
    attention_prefill: {
      ..args.attention_prefill,
      shared_memory_bytes: shared,
      module_path: path,
    },
    attention_prefill_wide: {
      ..args.attention_prefill_wide,
      shared_memory_bytes: shared,
      module_path: path,
    },
  }
  validate_reference_prefill_choice(args, ingress)
  (
    args,
    if vendor {
      measured.vendor_rows
    } else {
      0
    },
    attention * 2 + (if vendor { 1 } else { 0 }),
  )
}

///|
fn wb_target(
  minor? : Int = 0,
) -> @catalog.DeviceTarget raise @catalog.CatalogError {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn wb_claim(
  rank : Int,
  target : @catalog.DeviceTarget,
) -> TensorParallelRankDeclaration raise TensorParallelAdmissionError {
  TensorParallelRankDeclaration::new(
    rank~,
    world_size=2,
    process_ordinal=if rank == 0 { 2 } else { 5 },
    device_name=if rank == 0 { "gpu-two" } else { "gpu-five" },
    target~,
    peer_rank_count=1,
    memory_ceiling_bytes=1073741824L,
    weight_reserved_bytes=268435456L,
    activation_workspace_reserved_bytes=0L,
    kv_reserved_bytes=0L,
    collective_reserved_bytes=0L,
  )
}

///|
fn wb_declaration(
  target : @catalog.DeviceTarget,
) -> TensorParallelRuntimeDeclaration raise TensorParallelAdmissionError {
  TensorParallelRuntimeDeclaration::new(host_count=1, pipeline_stage_count=1, [
    wb_claim(0, target),
    wb_claim(1, target),
  ])
}

///|
fn wb_facts(
  target : @catalog.DeviceTarget,
  peer_link_count? : Int = 1,
) -> TopologyFacts {
  {
    driver_version: 12080,
    world_size: 2,
    target,
    peer_link_count,
    ranks: ReadOnlyArray::from_array([
      {
        rank: 0,
        process_ordinal: 2,
        device_name: "gpu-two",
        physical_bytes: 2147483648L,
        target,
      },
      {
        rank: 1,
        process_ordinal: 5,
        device_name: "gpu-five",
        physical_bytes: 2147483648L,
        target,
      },
    ]),
  }
}

///|
test "private topology seam rejects incomplete and heterogeneous evidence" {
  let target = wb_target()
  try
    admit_facts(wb_facts(target, peer_link_count=0), wb_declaration(target))
  catch {
    Rejected(area=Topology, reason=IncompletePeerLinks, rank=-1) => ()
    _ => fail("unexpected incomplete-topology rejection")
  } noraise {
    _ => fail("incomplete topology facts were admitted")
  }
  let other = wb_target(minor=6)
  let facts = wb_facts(target)
  let heterogeneous : TopologyFacts = {
    ..facts,
    ranks: ReadOnlyArray::from_array([
      facts.ranks[0],
      { ..facts.ranks[1], target: other },
    ]),
  }
  try admit_facts(heterogeneous, wb_declaration(target)) catch {
    Rejected(area=Topology, reason=TargetMismatch, rank=1) => ()
    _ => fail("unexpected heterogeneous-topology rejection")
  } noraise {
    _ => fail("heterogeneous topology facts were admitted")
  }
}

///|
test "device identities with report-control characters fail declaration" {
  let target = wb_target()
  try
    TensorParallelRankDeclaration::new(
      rank=0,
      world_size=2,
      process_ordinal=2,
      device_name="gpu\nforged",
      target~,
      peer_rank_count=1,
      memory_ceiling_bytes=1073741824L,
      weight_reserved_bytes=0L,
      activation_workspace_reserved_bytes=0L,
      kv_reserved_bytes=0L,
      collective_reserved_bytes=0L,
    )
  catch {
    Rejected(area=Declaration, reason=InvalidDeviceIdentity, rank=0) => ()
    _ => fail("unexpected unsafe-device-name rejection")
  } noraise {
    _ => fail("report-control device identity was admitted")
  }
}

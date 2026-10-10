///|
// The logical segmented manifest is not a resident whole-component arena.
// Still keep its maximum segment within both the caller's component budget
// and the signed-Int allocation ABI; tiny components must remain admissible.
fn component_host_limits(
  required : Int,
  maximum_component_bytes : Int64,
) -> @host.MiniMaxH3HostMaterializationLimits raise {
  @host.MiniMaxH3HostMaterializationLimits::new(
    maximum_shards=64,
    maximum_tensors=required,
    maximum_header_bytes=4194304,
    maximum_payload_bytes_per_shard=maximum_component_bytes,
    maximum_total_host_bytes=maximum_component_bytes,
    maximum_arena_bytes=Int64::min(2147483584L, maximum_component_bytes),
    maximum_arenas=256,
    maximum_name_bytes=512,
  )
}

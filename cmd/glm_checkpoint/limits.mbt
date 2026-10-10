///|
// Sharding is a storage choice, not model semantics. The same bounded total
// tensor population must fit whether scales are spread across weight shards or
// concentrated in one ModelOpt auxiliary shard. Header/file byte bounds remain
// independent of this count limit.
fn checkpoint_limits(
  maximum_copy_bytes : Int64,
) -> @streaming.StreamingSafetensorsLimits raise {
  @streaming.StreamingSafetensorsLimits::new(
    maximum_shards=512,
    maximum_file_bytes=17179869184L,
    maximum_header_bytes=33554432,
    maximum_tensors_per_shard=500000,
    maximum_total_tensors=500000,
    maximum_rank=8,
    maximum_name_bytes=512,
    maximum_dtype_specs=9,
    maximum_copy_requests=20000,
    maximum_copy_bytes~,
  )
}

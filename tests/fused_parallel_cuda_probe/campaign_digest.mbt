///|
fn campaign_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn campaign_digest(
  value : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(value) catch {
    _ =>
      raise FusedCudaProbeRejected("campaign digest is not canonical SHA-256")
  }
}

///|
fn campaign_referee_shape_set_digest() -> String {
  let output = StringBuilder()
  output.write_string("schema=lunaflux-fused-scalar-referee-shape-set.v1\n")
  let shapes = @fused_parallel_qualification.declared_qualification_shapes()
  output.write_string("shape_count=" + shapes.length().to_string() + "\n")
  for shape in shapes {
    output.write_string(
      "shape=" +
      shape.ordinal().to_string() +
      "," +
      shape.name() +
      "," +
      [ for value in shape.positions() => value.to_string() ].join("|") +
      "," +
      [ for value in shape.page_table() => value.to_string() ].join("|") +
      "\n",
    )
  }
  campaign_sha256(@utf8.encode(output.to_string()))
}

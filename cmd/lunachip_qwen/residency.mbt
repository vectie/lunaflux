///|
// The final 64 KiB is outside the admitted layout. Keep two SHA-256 digests
// here. The reset-sensitive CSR at 0x29c ties the record to live DDR.
let record_address : UInt64 = 0x7fff0000UL

///|
priv struct ResidentIdentity {
  weights : Bytes
  programs : Bytes
  checkpoint : String
  source_stamp : @approved.ApprovedFileStamp
}

///|
fn resident_identity(
  source : Source,
  plans : Array[@qwen.DecodeStep],
) -> ResidentIdentity raise {
  let hash = @crypto.SHA256::new()
  let stamp = source.file.stamp()
  let mut offset = 0L
  while offset < stamp.byte_count() {
    let n = (stamp.byte_count() - offset).min(1048576L).to_int()
    hash.update(read_range(source.file, offset, n))
    offset += n.to_int64()
  }
  assert_true(source.file.stamp() == stamp)
  let digest = hash.finalize()
  let weights = @crypto.SHA256::new()
  weights.update(b"LunaFlux PBlaze3 BF16 weights and expanded FP32 norms v1")
  weights.update(digest)
  for t in source.transfers {
    for
      v in [
        t.source_offset.reinterpret_as_uint64(),
        t.source_bytes.reinterpret_as_uint64(),
        t.destination.address,
        match t.expanded_norm {
          Some(r) => r.address
          None => 0UL
        },
      ] {
      weights.update(words_bytes([v.to_uint(), (v >> 32).to_uint()]))
    }
  }
  let program = @crypto.SHA256::new()
  program.update(b"LunaFlux packed resident program v1")
  let workspace = source.layout.workspace()
  assert_true(
    source.layout.reserve().address + source.layout.reserve().bytes <=
    record_address,
  )
  program.update(
    words_bytes([
      workspace.address.to_uint(),
      plans.length().reinterpret_as_uint(),
    ]),
  )
  for plan in plans {
    program.update(words_bytes([plan.program_count().reinterpret_as_uint()]))
    for layer in 0..<plan.program_count() {
      let p = plan.program(layer).unwrap()
      program.update(words_bytes([p.chunk_count().reinterpret_as_uint()]))
      for c in 0..<p.chunk_count() {
        let bytes = p.chunk(c).unwrap()
        program.update(words_bytes([bytes.length().reinterpret_as_uint()]))
        program.update(bytes)
      }
    }
  }
  {
    weights: Bytes::from_array(weights.finalize()[:]),
    programs: Bytes::from_array(program.finalize()[:]),
    checkpoint: @crypto.bytes_to_hex_string(digest),
    source_stamp: stamp,
  }
}

///|
async fn Jtag::resident_matches(
  self : Self,
  id : ResidentIdentity,
) -> (Bool, Bool) {
  assert_eq(
    self.request("DIGEST"),
    id.checkpoint,
    msg="bridge checkpoint payload differs",
  )
  if self.word(0x8010029cUL) != 1U {
    return (false, false)
  }
  let weights = self.read(record_address, 32) == id.weights
  (weights, weights && self.read(record_address + 32UL, 32) == id.programs)
}

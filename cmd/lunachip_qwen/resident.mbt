///|
priv struct ResidentProgram {
  address : UInt64
  count : Int32
} derive(ToJson)

///|
priv struct CycleSnapshot {
  sequence : UInt32
  status : UInt32
  error : UInt32
  count : UInt32
  core_cycles : UInt64
  loader_cycles : UInt64
  instructions : UInt32
  products : UInt32
  ddr_beats : UInt32
  completed_segments : UInt32?
  chain_cycles : UInt32?
  after : UInt32
  selected_token : UInt32?
} derive(ToJson)

///|
fn cycle_snapshot(
  bytes : Bytes,
  chain~ : Bool = false,
  greedy~ : Bool = false,
) -> CycleSnapshot raise {
  guard (!greedy || chain) &&
    bytes.length() == (if greedy { 68 } else if chain { 64 } else { 56 }) else {
    fail("invalid counter snapshot")
  }
  {
    sequence: decode_word(bytes, 0),
    status: decode_word(bytes, 1),
    error: decode_word(bytes, 2),
    count: decode_word(bytes, 3),
    core_cycles: decode_word(bytes, 7).to_uint64() |
    (decode_word(bytes, 8).to_uint64() << 32),
    loader_cycles: decode_word(bytes, 12).to_uint64(),
    instructions: decode_word(bytes, 9),
    products: decode_word(bytes, 10),
    ddr_beats: decode_word(bytes, 11),
    completed_segments: if chain {
      Some(decode_word(bytes, 14))
    } else {
      None
    },
    chain_cycles: if chain {
      Some(decode_word(bytes, 15))
    } else {
      None
    },
    after: decode_word(bytes, 13),
    selected_token: if greedy {
      let id = decode_word(bytes, 16)
      guard id < 151936U else { fail("invalid FPGA argmax") }
      Some(id)
    } else {
      None
    },
  }
}

///|
async fn Jtag::resident_execute(
  self : Self,
  program : ResidentProgram,
  previous : UInt32,
  chain~ : Bool = false,
  token_setup~ : (UInt64, UInt64, Bytes)? = None,
) -> CycleSnapshot {
  let greedy = token_setup is Some((_, _, bytes)) &&
    bytes.length() == 4 &&
    @env.get_env_var("PBLAZE3_GREEDY_QUEUE") == Some("1")
  let request = match token_setup {
    Some((logits, embedding, bytes)) => {
      guard chain && (bytes.length() == 4096 || bytes.length() == 4) else {
        fail("invalid queued token")
      }
      let opcode = if greedy {
        "G"
      } else if bytes.length() == 4 {
        "E"
      } else {
        "T"
      }
      "\{opcode} \{program.address} \{program.count} \{previous} \{logits} \{embedding} \{encode_hex(bytes[:])}"
    }
    None => {
      let opcode = if chain { "C" } else { "X" }
      "\{opcode} \{program.address} \{program.count} \{previous}"
    }
  }
  ignore(self.owner.begin(previous, program.count, chain~))
  try {
    let snapshot = cycle_snapshot(
      decode_hex(self.request(request, timeout_ms=610000)),
      chain~,
      greedy~,
    )
    assert_true(
      self.owner.observe(
        snapshot.sequence,
        snapshot.after,
        snapshot.status,
        snapshot.error,
        snapshot.count,
      ),
      msg="incomplete resident launch",
    )
    assert_true(snapshot.core_cycles > 0UL && snapshot.loader_cycles > 0UL)
    if chain {
      let segments = ((program.count + 2047) / 2048).to_uint()
      assert_eq(snapshot.completed_segments, Some(segments))
      assert_eq(snapshot.instructions + segments, snapshot.count)
      assert_eq(
        snapshot.chain_cycles.unwrap().to_uint64(),
        snapshot.core_cycles + snapshot.loader_cycles + segments.to_uint64(),
      )
    }
    snapshot
  } catch {
    e => {
      self.owner.invalidate()
      raise e
    }
  }
}

///|
async fn Jtag::upload_weights(
  self : Self,
  source : Source,
  output : String,
  verify~ : Bool = true,
) -> Int64 {
  let operation = if verify { "F" } else { "U" }
  let mut uploaded = 0L
  for t in source.transfers {
    let mut offset = 0L
    while t.chunk(offset, 1048576) is Some(chunk) {
      ignore(
        self.request(
          "\{operation} \{chunk.destination_address} \{source.payload + chunk.source_offset} \{chunk.source_bytes}",
        ),
      )
      offset += chunk.source_bytes.to_int64()
      uploaded += chunk.source_bytes.to_int64()
      @fs.write_file(
        output + "/upload-progress.json",
        Json::object({
          "uploaded_bytes": uploaded.to_json(),
          "full_weight_readback": verify.to_json(),
          "total_bytes": 1192099840L.to_json(),
          "tensor": t.ordinal.to_json(),
        }).stringify(),
      )
    }
    if t.expanded_norm is Some(region) {
      let raw = source.weight(t.ordinal)
      let values = Array::make(raw.length() * 2, b'\x00')
      @qwen.expand_bf16(raw[:], values)
      self.write(region.address, Bytes::from_array(values[:]))
    }
    println("Uploaded \{uploaded}/1192099840 bytes (tensor \{t.ordinal})")
  }
  assert_eq(uploaded, 1192099840L)
  uploaded
}

///|
/// Program bytes occupy a distinct tail of the reserved workspace, after all
/// live tensors/logits. Each upload is read back once before any timed launch.
async fn Jtag::upload_programs(
  self : Self,
  source : Source,
  plans : Array[@qwen.DecodeStep],
  output : String,
  reuse~ : Bool = false,
) -> Array[Array[ResidentProgram]] {
  let workspace = source.layout.workspace()
  let mut address = workspace.address + 1048576UL
  let result : Array[Array[ResidentProgram]] = []
  for plan in plans {
    let programs : Array[ResidentProgram] = []
    for layer in 0..<plan.program_count() {
      let program = plan.program(layer).unwrap()
      for c in 0..<program.chunk_count() {
        let bytes = program.chunk(c).unwrap()
        assert_true(
          address + bytes.length().to_uint64() <=
          workspace.address + workspace.bytes,
          msg="resident programs exceed workspace",
        )
        if !reuse {
          self.write(address, bytes)
        }
        programs.push({ address, count: bytes.length() / 64, })
        address += bytes.length().to_uint64()
      }
    }
    result.push(programs)
  }
  @fs.write_file(
    output + "/resident-programs.json",
    result.to_json().stringify(indent=2),
    create_mode=CreateNew,
  )
  result
}

///|
/// Validate every boundary before timing, including the exclusive DDR end.
fn packed_chain(chunks : Array[ResidentProgram]) -> ResidentProgram raise {
  guard chunks.length() > 0 else { fail("empty resident chain") }
  let base = chunks[0].address
  guard base >= 64UL && base < 0x80000000UL && base % 64UL == 0UL else {
    fail("invalid resident base")
  }
  let mut count = 0
  for i, chunk in chunks {
    guard chunk.address == base + count.to_uint64() * 64UL &&
      chunk.count >= 1 &&
      chunk.count <= 2048 &&
      (i + 1 == chunks.length() || chunk.count == 2048) else {
      fail("unpacked resident chain")
    }
    count += chunk.count
    guard count <= 65535 && base + count.to_uint64() * 64UL <= 0x80000000UL else {
      fail("resident chain exceeds aperture or counter")
    }
  }
  { address: base, count, }
}

///|
priv struct Jtag {
  connection : @socket.Tcp
  mut sequence : Int32
  owner : @device.LaunchOwner
}

///|
fn decode_hex(hex : StringView) -> Bytes raise {
  guard hex.length() % 2 == 0 &&
    hex.iter().all(c => c is ('0'..<='9' | 'a'..<='f' | 'A'..<='F')) else {
    fail("invalid hex response")
  }
  let output = Array::make(hex.length() / 2, b'\x00')
  for i in 0..<output.length() {
    output[i] = @string.parse_uint(hex[i * 2:i * 2 + 2], base=16).to_byte()
  }
  Bytes::from_array(output[:])
}

///|
fn encode_hex(bytes : BytesView) -> String {
  let text = StringBuilder(size_hint=bytes.length() * 2)
  let digits = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
  ]
  for byte in bytes {
    text.write_char(digits[byte.to_int() >> 4])
    text.write_char(digits[byte.to_int() & 15])
  }
  text.to_string()
}

///|
async fn Jtag::request(
  self : Self,
  request : String,
  timeout_ms~ : Int32 = 120000,
) -> String {
  self.sequence += 1
  @async.with_timeout(timeout_ms, () => {
    self.connection.write("\{self.sequence} \{request}\n")
    guard self.connection.read_until("\n") is Some(line) else {
      fail("board transport closed")
    }
    let prefix = "\{self.sequence} OK "
    guard line.has_prefix(prefix) else { fail("board request failed: \{line}") }
    line[prefix.length():].trim_end().to_owned()
  }) catch {
    e => {
      self.owner.invalidate()
      raise e
    }
  }
}

///|
async fn Jtag::write(self : Self, address : UInt64, bytes : Bytes) -> Unit {
  let mut offset = 0
  while offset < bytes.length() {
    let count = (bytes.length() - offset).min(65536)
    ignore(
      self.request(
        "W \{address + offset.to_uint64()} \{encode_hex(bytes[offset:offset + count])}",
      ),
    )
    offset += count
  }
}

///|
async fn Jtag::read(self : Self, address : UInt64, count : Int32) -> Bytes {
  try {
    let bytes = decode_hex(self.request("R \{address} \{count}"))
    assert_eq(bytes.length(), count)
    bytes
  } catch {
    e => {
      self.owner.invalidate()
      raise e
    }
  }
}

///|
async fn Jtag::word(self : Self, address : UInt64) -> UInt32 {
  decode_word(self.read(address, 4), 0)
}

///|
async fn Jtag::set_word(self : Self, address : UInt64, word : UInt32) -> Unit {
  self.write(address, float_bytes([Float::reinterpret_from_uint(word)]))
}

///|
async fn Jtag::execute(self : Self, program : @device.PreparedProgram) -> Unit {
  for c in 0..<program.chunk_count() {
    self.execute_chunk(program.chunk(c).unwrap())
  }
}

///|
async fn Jtag::execute_chunk(self : Self, bytes : Bytes) -> Unit {
  let base = 0x80100000UL
  let status = self.word(base + 8UL)
  assert_true(
    (status & 11U) == 1U,
    msg="device not healthy and idle before upload",
  )
  self.write(base + 0x20000UL, bytes)
  self.set_word(base + 0x14UL, (bytes.length() / 64).to_uint())
  ignore(self.owner.begin(self.word(base + 0x38UL), bytes.length() / 64))
  try {
    self.set_word(base + 0x10UL, 1U)
    @async.with_timeout(600000, () => {
      while true {
        let before = self.word(base + 0x38UL)
        let status = self.word(base + 8UL)
        let error = self.word(base + 12UL)
        let count = self.word(base + 0x14UL)
        let after = self.word(base + 0x38UL)
        if self.owner.observe(before, after, status, error, count) {
          break
        }
        @async.sleep(10)
      }
    })
  } catch {
    e => {
      self.owner.invalidate()
      raise e
    }
  }
}

///|
/// Experimental loopback transport. It requires an already qualified image and
/// a single-use vendor bridge; no reconnect/retry or partial-upload reuse.
async fn physical(
  source : Source,
  plans : Array[@qwen.DecodeStep],
  tokenizer : @tokenizer.TokenizerSpec,
  tokens : Array[Int32],
  generation : Int32,
  output : String,
  endpoint : String,
  qualification : String,
) -> Unit {
  guard endpoint.has_prefix("127.0.0.1:") else {
    fail("SSH loopback endpoint required")
  }
  let connection = @socket.Tcp::connect(@socket.Addr::parse(endpoint))
  defer connection.close()
  let jtag = Jtag::{
    connection,
    sequence: 0,
    owner: @device.LaunchOwner::new(),
  }
  let remote_meta = jtag.request("META").split(" ").to_array()
  assert_eq(remote_meta.length(), 2)
  assert_eq(
    @string.parse_int64(remote_meta[0]),
    source.file.stamp().byte_count(),
  )
  assert_eq(
    decode_hex(remote_meta[1]),
    read_range(source.file, 0L, source.payload.to_int()),
  )
  assert_eq(jtag.word(0x80000000UL), 15U)
  assert_eq(jtag.word(0x80100000UL), 0x50423330U)
  assert_eq(jtag.word(0x80100004UL), 0x00010001U)
  assert_eq(jtag.word(0x80100080UL), 0x80000000U)
  assert_eq(jtag.word(0x80100084UL), 0U)
  assert_eq(jtag.word(0x80100088UL), 31U)
  jtag.qualify(qualification, output)
  // Probe a nonzero asymmetric pattern and its inverse across a 4 KiB edge
  // before the multi-GiB write/readback campaign.
  let probe = Array::makei(8192, i => ((i * 37 + i / 251 + 19) % 256).to_byte())
  jtag.write(
    source.layout.reserve().address + 3584UL,
    Bytes::from_array(probe[:]),
  )
  assert_eq(
    jtag.read(source.layout.reserve().address + 3584UL, 8192),
    Bytes::from_array(probe[:]),
  )
  let uploaded = jtag.upload_weights(source, output)
  let emulator = Emulator::new(source)
  let reference = Reference::new(source)
  let prompt_count = tokens.length()
  let results : Array[Json] = []
  for position in 0..<(prompt_count + generation - 1) {
    let plan = plans[position]
    let token = tokens[position]
    let embedding = float_bytes(source.embedding(token)[:])
    let constants = plan.constants()
    jtag.write(plan.embedding_address(), embedding)
    jtag.write(plan.constants_address(), constants)
    emulator.write(plan.embedding_address(), embedding)
    emulator.write(plan.constants_address(), constants)
    for layer in 0..<plan.program_count() {
      jtag.execute(plan.program(layer).unwrap())
      emulator.execute(plan.program(layer).unwrap())
      println("FPGA completed position \{position} layer/program \{layer}")
    }
    let physical_bytes = jtag.read(plan.logits_address(), 151936 * 4)
    @fs.write_file(
      output + "/p\{position}-physical-logits.bin",
      physical_bytes,
      create_mode=CreateNew,
    )
    let actual = Array::makei(151936, i => {
      Float::reinterpret_from_uint(decode_word(physical_bytes, i))
    })
    let golden = Array::makei(151936, i => {
      emulator.read(plan.logits_address() + i.to_uint64() * 4UL)
    })
    @fs.write_file(
      output + "/p\{position}-emulator-logits.bin",
      float_bytes(golden[:]),
      create_mode=CreateNew,
    )
    assert_true(
      physical_bytes == float_bytes(golden[:]),
      msg="FPGA/ISA full logits differ; binary outputs retained",
    )
    let expected = reference.step(token, position)
    @fs.write_file(
      output + "/p\{position}-reference-logits.bin",
      float_bytes(expected[:]),
      create_mode=CreateNew,
    )
    let actual_token = greedy(actual[:])
    assert_eq(
      actual_token,
      greedy(expected[:]),
      msg="independent reference token differs",
    )
    let mut max_error = 0.0
    for i in 0..<151936 {
      max_error = max_error.max(
        (actual[i].to_double() - expected[i].to_double()).abs(),
      )
    }
    assert_true(max_error <= 0.05)
    results.push({
      "position": position,
      "input_token": token,
      "generated_token": actual_token,
      "exact_fpga_isa_logit_words": 151936,
      "reference_max_absolute_error": max_error,
    })
    @fs.write_file(
      output + "/physical-progress.json",
      Json::array(results.copy()).stringify(indent=2),
    )
    if position >= prompt_count - 1 {
      tokens.push(actual_token)
      if terminal_token(actual_token) {
        break
      }
    }
  }
  let result : Json = {
    "schema": "lunaflux.pblaze3-full-qwen-physical/v1",
    "physical_execution": true,
    "weight_bytes_uploaded_and_readback_verified": uploaded,
    "positions": results,
    "tokens": tokens,
    "generated_text": tokenizer.decode_text(
      tokens[prompt_count:],
      special_policy=Preserve,
    ),
    "transport": "single-owner JTAG",
    "numeric_profile": "qwen-layer-ordered-fp32-approx-v1",
  }
  @fs.write_file(
    output + "/physical-result.json",
    result.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(result.stringify(indent=2))
}

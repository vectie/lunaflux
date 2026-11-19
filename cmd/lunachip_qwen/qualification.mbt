///|
priv struct QualificationMemory {
  address : UInt64
  words : Array[UInt32]
} derive(FromJson)

///|
priv struct QualificationFixture {
  schema : String
  source_program : String
  program_words : Array[UInt32]
  assets : Array[QualificationMemory]
  outputs : Array[QualificationMemory]
} derive(FromJson)

///|
fn words_bytes(words : ArrayView[UInt32]) -> Bytes {
  let bytes = Array::make(words.length() * 4, b'\x00')
  for i, word in words {
    for j in 0..<4 {
      bytes[i * 4 + j] = (word >> (j * 8)).to_byte()
    }
  }
  Bytes::from_array(bytes[:])
}

///|
async fn Jtag::qualify(
  self : Self,
  path : String,
  output : String,
  resident~ : UInt64 = 0UL,
  chain~ : Bool = false,
) -> Unit {
  guard @fs.file_size(path) <= 1048576L else {
    fail("qualification fixture too large")
  }
  let raw = @fs.read_file(path).text()
  let fixture : QualificationFixture = @json.from_json(@json.parse(raw))
  assert_eq(fixture.schema, "lunachip.adapter-fixture/v1")
  let bytes = words_bytes(fixture.program_words[:])
  assert_true(
    bytes.length() >= 128 &&
    bytes.length() <= 131072 &&
    bytes.length() % 64 == 0,
  )
  assert_eq(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    fixture.source_program,
  )
  for i in 0..<(fixture.program_words.length() / 16) {
    assert_true(fixture.program_words[i * 16] <= 15U)
    assert_eq(
      fixture.program_words[i * 16] == 0U,
      i == fixture.program_words.length() / 16 - 1,
    )
  }
  let guards : Array[(UInt64, Bytes)] = []
  for memory in fixture.assets {
    assert_true(
      memory.address < 0x80000000UL &&
      memory.address % 4UL == 0UL &&
      memory.words.length() <= 4096 &&
      !memory.words.is_empty() &&
      memory.address + memory.words.length().to_uint64() * 4UL <= 0x80000000UL,
    )
    self.write(memory.address, words_bytes(memory.words[:]))
  }
  for memory in fixture.outputs {
    assert_true(
      memory.address >= 4UL &&
      memory.address < 0x80000000UL &&
      memory.address % 4UL == 0UL &&
      memory.words.length() <= 4096 &&
      !memory.words.is_empty() &&
      memory.address + memory.words.length().to_uint64() * 4UL + 4UL <=
      0x80000000UL,
    )
    let end = memory.address + memory.words.length().to_uint64() * 4UL
    let sentinel = b"\xa5\x17\xce\x42"
    self.write(memory.address - 4UL, sentinel)
    self.write(end, sentinel)
    guards.push((memory.address - 4UL, sentinel))
    guards.push((end, sentinel))
    self.write(
      memory.address,
      words_bytes(Array::make(memory.words.length(), 0x7fc00000U)[:]),
    )
  }
  @fs.write_file(
    output + "/qualification-fixture.json",
    raw,
    create_mode=CreateNew,
  )
  let cycles = if resident == 0UL {
    self.execute_chunk(bytes)
    None
  } else {
    // Exercise a real segment boundary before the long checkpoint upload.
    // An elementwise add of zero touches one reserved scratch word only.
    let staged = if chain {
      let prefix = Array::make(2048 * 16, 0U)
      let add = [
        2U, 30000U, 30000U, 0U, 1U, 0U, 1U, 0U, 0U, 0U, 2U, 0U, 0U, 0U, 0U, 0U,
      ]
      for i in 0..<2047 {
        for j, word in add {
          prefix[i * 16 + j] = word
        }
      }
      for word in fixture.program_words {
        prefix.push(word)
      }
      words_bytes(prefix[:])
    } else {
      bytes
    }
    self.write(resident, staged)
    let snapshot = self.resident_execute(
      { address: resident, count: staged.length() / 64, },
      self.word(0x80100038UL),
      chain~,
    )
    self.set_word(0x8010003cUL, 0U)
    Some(snapshot)
  }
  let observed : Array[Json] = []
  let mut checked = 0
  for memory in fixture.outputs {
    let actual = self.read(memory.address, memory.words.length() * 4)
    observed.push({
      "address": memory.address,
      "data_hex": @crypto.bytes_to_hex_string(actual),
    })
    @fs.write_file(
      output + "/qualification-progress.json",
      Json::array(observed.copy()).stringify(indent=2),
    )
    assert_true(
      actual == words_bytes(memory.words[:]),
      msg="physical operator qualification failed",
    )
    checked += memory.words.length()
  }
  for (address, expected) in guards {
    assert_eq(self.read(address, 4), expected)
  }
  @fs.write_file(
    output + "/qualification-result.json",
    (
      {
        "status": "PASS",
        "physical_execution": true,
        "checked_words": checked,
        "guard_words": guards.length(),
        "source_program": fixture.source_program,
        "resident_program_address": resident,
        "cycles": cycles,
      } : Json).stringify(indent=2),
    create_mode=CreateNew,
  )
  println("Physical operator qualification PASS: \{checked} exact FP32 words")
}

///|
priv struct KvWriteCase {
  name : String
  step_counts : ReadOnlyArray[Int]
  positions : ReadOnlyArray[Int]
  row_offsets : ReadOnlyArray[Int]
  sequence_lengths : ReadOnlyArray[Int]
  page_offsets : ReadOnlyArray[Int]
  pages : ReadOnlyArray[Int]
}

///|
fn kv_case(
  name : String,
  prefill : Int,
  decode : Int,
  positions : Array[Int],
  row_offsets : Array[Int],
  sequence_lengths : Array[Int],
  page_offsets : Array[Int],
  pages : Array[Int],
) -> KvWriteCase {
  {
    name,
    step_counts: [
      prefill,
      decode,
      prefill + decode,
      positions.length(),
      pages.length(),
    ],
    positions: ReadOnlyArray::from_array(positions),
    row_offsets: ReadOnlyArray::from_array(row_offsets),
    sequence_lengths: ReadOnlyArray::from_array(sequence_lengths),
    page_offsets: ReadOnlyArray::from_array(page_offsets),
    pages: ReadOnlyArray::from_array(pages),
  }
}

///|
fn kv_probe_cases() -> ReadOnlyArray[KvWriteCase] {
  [
    kv_case("origin-prefill", 1, 0, [0], [0, 1], [1], [0, 1], [1]),
    kv_case("page-tail-decode", 0, 1, [15], [0, 1], [16], [0, 1], [2]),
    kv_case("cross-page-prefill", 1, 0, [15, 16], [0, 2], [17], [0, 2], [1, 3]),
    kv_case("multirow-decode", 0, 2, [0, 15], [0, 1, 2], [1, 16], [0, 1, 2], [
      4, 5,
    ]),
    kv_case(
      "mixed-prefill-decode",
      1,
      1,
      [14, 15, 16],
      [0, 2, 3],
      [16, 17],
      [0, 1, 3],
      [6, 7, 8],
    ),
    kv_case(
      "full-grid-mixed",
      2,
      2,
      [14, 15, 16, 14, 15, 16, 16, 31],
      [0, 3, 6, 7, 8],
      [17, 17, 17, 32],
      [0, 2, 4, 6, 8],
      [9, 10, 11, 12, 13, 14, 15, 16],
    ),
  ]
}

///|
fn kv_case_row(case_ : KvWriteCase, token : Int) -> Int {
  let mut row = 0
  while row + 1 < case_.step_counts[2] && token >= case_.row_offsets[row + 1] {
    row += 1
  }
  row
}

///|
fn kv_case_qkv_bits(case_ : KvWriteCase) -> Array[Int] {
  Array::makei(case_.positions.length() * 256, index => {
    0x3c00 + (case_.name.length() * 257 + index * 13) % 0x0200
  })
}

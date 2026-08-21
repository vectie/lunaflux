///|
fn luna_semantic_utf8_width(codepoint : Int) -> Int {
  if codepoint <= 0x7f {
    1
  } else if codepoint <= 0x7ff {
    2
  } else if codepoint <= 0xffff {
    3
  } else {
    4
  }
}

///|
fn luna_semantic_utf8_byte(codepoint : Int, width : Int, index : Int) -> Byte {
  if width == 1 {
    codepoint.to_byte()
  } else if index == 0 {
    ((0xf0 << (4 - width)) | (codepoint >> (6 * (width - 1)))).to_byte()
  } else {
    (0x80 | ((codepoint >> (6 * (width - index - 1))) & 0x3f)).to_byte()
  }
}

///|
fn luna_semantic_codepoint_at(value : String, offset : Int) -> Int {
  let first = value.unsafe_get(offset).to_int()
  if first >= 0xd800 && first <= 0xdbff {
    if offset + 1 >= value.length() {
      abort("validated stop string surrogate invariant")
    }
    let second = value.unsafe_get(offset + 1).to_int()
    if second < 0xdc00 || second > 0xdfff {
      abort("validated stop string surrogate invariant")
    }
    0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
  } else {
    if first >= 0xdc00 && first <= 0xdfff {
      abort("validated stop string surrogate invariant")
    }
    first
  }
}

///|
fn luna_semantic_codepoint_units_at(value : String, offset : Int) -> Int {
  let first = value.unsafe_get(offset).to_int()
  if first >= 0xd800 && first <= 0xdbff {
    2
  } else {
    1
  }
}

///|
fn LunaRequestPreparationLane::require_semantic_write(
  self : LunaRequestPreparationLane,
) -> @inference.LunaRequestSemanticWrite {
  if self.semantic_write.length() != 1 {
    abort("Luna semantic write invariant")
  }
  self.semantic_write[0]
}

///|
fn LunaRequestPreparationLane::require_semantic_work(
  self : LunaRequestPreparationLane,
) -> @inference.LunaRequestSemanticWork {
  if self.semantic_work.length() != 1 {
    abort("Luna semantic work invariant")
  }
  self.semantic_work[0]
}

///|
fn LunaRequestPreparationLane::require_semantic_lease(
  self : LunaRequestPreparationLane,
) -> @inference.LunaRequestSemanticLease {
  if self.semantic_lease.length() != 1 {
    abort("Luna semantic lease invariant")
  }
  self.semantic_lease[0]
}

///|
fn LunaRequestPreparationLane::begin_semantic_import(
  self : LunaRequestPreparationLane,
) -> Unit {
  self.semantic_item_index = 0
  self.semantic_unit_offset = 0
  self.semantic_byte_length = 0
  self.semantic_codepoint = 0
  self.semantic_codepoint_width = 0
  self.semantic_codepoint_byte_index = 0
  self.phase = LUNA_PREPARATION_SEMANTIC_CACHE_MEASURE
}

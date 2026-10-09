///|
pub(all) suberror IndexedParameterBankLayoutError {
  InvalidAlignment
  InvalidExtent
  DuplicateSourceOrdinal(Int)
  DuplicateSourceName(String)
  UnexpectedLogicalIndex(expected~ : Int, actual~ : Int)
  OffsetOverflow
} derive(Debug, Eq)

///|
pub struct IndexedParameterSliceInput {
  priv logical_index : Int
  priv source_ordinal : Int
  priv source_name : String
  priv byte_count : Int64
}

///|
pub fn IndexedParameterSliceInput::new(
  logical_index~ : Int,
  source_ordinal~ : Int,
  source_name~ : String,
  byte_count~ : Int64,
) -> Self raise IndexedParameterBankLayoutError {
  if logical_index < 0 || source_ordinal < 0 || source_name.is_empty() {
    raise InvalidExtent
  }
  { logical_index, source_ordinal, source_name, byte_count, }
}

///|
pub struct IndexedParameterSlice {
  priv logical_index : Int
  priv source_ordinal : Int
  priv source_name : String
  priv target_byte_offset : Int64
  priv byte_count : Int64
}

///|
pub struct IndexedParameterBankLayout {
  priv alignment : Int64
  priv byte_count : Int64
  priv slices : ReadOnlyArray[IndexedParameterSlice]
  priv digest : IndexedParameterBankLayoutDigest
}

///|
pub struct IndexedParameterBankLayoutDigest(String) derive(Debug, Eq, Hash)

///|
pub fn build_indexed_parameter_bank_layout(
  inputs : ArrayView[IndexedParameterSliceInput],
  alignment~ : Int64,
  maximum_byte_count~ : Int64,
) -> IndexedParameterBankLayout raise IndexedParameterBankLayoutError {
  if alignment <= 0L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L {
    raise InvalidAlignment
  }
  if inputs.is_empty() || maximum_byte_count <= 0L {
    raise InvalidExtent
  }
  let slices : Array[IndexedParameterSlice] = []
  let mut offset = 0L
  for index, input in inputs {
    if input.logical_index != index {
      raise UnexpectedLogicalIndex(expected=index, actual=input.logical_index)
    }
    if input.byte_count <= 0L || input.byte_count % alignment != 0L {
      raise InvalidExtent
    }
    for previous in slices {
      if previous.source_ordinal == input.source_ordinal {
        raise DuplicateSourceOrdinal(input.source_ordinal)
      }
      if previous.source_name == input.source_name {
        raise DuplicateSourceName(input.source_name)
      }
    }
    if offset > maximum_byte_count - input.byte_count {
      raise OffsetOverflow
    }
    slices.push({
      logical_index: input.logical_index,
      source_ordinal: input.source_ordinal,
      source_name: input.source_name,
      target_byte_offset: offset,
      byte_count: input.byte_count,
    })
    offset += input.byte_count
  }
  let canonical = StringBuilder()
  canonical.write_string("lunaflux.indexed-parameter-bank-layout.v1|")
  canonical.write_string(alignment.to_string())
  canonical.write_char('|')
  canonical.write_string(offset.to_string())
  for item in slices {
    canonical.write_char('|')
    canonical.write_string(item.logical_index.to_string())
    canonical.write_char(':')
    canonical.write_string(item.source_ordinal.to_string())
    canonical.write_char(':')
    canonical.write_string(item.source_name.length().to_string())
    canonical.write_char(':')
    canonical.write_string(item.source_name)
    canonical.write_char(':')
    canonical.write_string(item.target_byte_offset.to_string())
    canonical.write_char(':')
    canonical.write_string(item.byte_count.to_string())
  }
  let digest = IndexedParameterBankLayoutDigest(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(canonical.to_string())),
    ),
  )
  {
    alignment,
    byte_count: offset,
    slices: ReadOnlyArray::from_array(slices),
    digest,
  }
}

///|
pub fn IndexedParameterBankLayout::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn IndexedParameterBankLayout::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn IndexedParameterBankLayout::slices(
  self : Self,
) -> ReadOnlyArray[IndexedParameterSlice] {
  self.slices
}

///|
pub fn IndexedParameterBankLayout::digest(
  self : Self,
) -> IndexedParameterBankLayoutDigest {
  self.digest
}

///|
pub fn IndexedParameterBankLayoutDigest::as_hex(self : Self) -> String {
  self.0
}

///|
pub fn IndexedParameterSlice::logical_index(self : Self) -> Int {
  self.logical_index
}

///|
pub fn IndexedParameterSlice::source_ordinal(self : Self) -> Int {
  self.source_ordinal
}

///|
pub fn IndexedParameterSlice::source_name(self : Self) -> String {
  self.source_name
}

///|
pub fn IndexedParameterSlice::target_byte_offset(self : Self) -> Int64 {
  self.target_byte_offset
}

///|
pub fn IndexedParameterSlice::byte_count(self : Self) -> Int64 {
  self.byte_count
}

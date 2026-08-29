///|
fn luna_http1_header_name_byte(kind : Int, index : Int) -> Byte {
  match kind {
    LUNA_HTTP1_HEADER_AUTHORIZATION =>
      match index {
        0 => b'a'
        1 => b'u'
        2 => b't'
        3 => b'h'
        4 => b'o'
        5 => b'r'
        6 => b'i'
        7 => b'z'
        8 => b'a'
        9 => b't'
        10 => b'i'
        11 => b'o'
        _ => b'n'
      }
    LUNA_HTTP1_HEADER_CONTENT_LENGTH =>
      match index {
        0 => b'c'
        1 => b'o'
        2 => b'n'
        3 => b't'
        4 => b'e'
        5 => b'n'
        6 => b't'
        7 => b'-'
        8 => b'l'
        9 => b'e'
        10 => b'n'
        11 => b'g'
        12 => b't'
        _ => b'h'
      }
    LUNA_HTTP1_HEADER_CONTENT_TYPE =>
      match index {
        0 => b'c'
        1 => b'o'
        2 => b'n'
        3 => b't'
        4 => b'e'
        5 => b'n'
        6 => b't'
        7 => b'-'
        8 => b't'
        9 => b'y'
        10 => b'p'
        _ => b'e'
      }
    _ =>
      match index {
        0 => b't'
        1 => b'r'
        2 => b'a'
        3 => b'n'
        4 => b's'
        5 => b'f'
        6 => b'e'
        7 => b'r'
        8 => b'-'
        9 => b'e'
        10 => b'n'
        11 => b'c'
        12 => b'o'
        13 => b'd'
        14 => b'i'
        15 => b'n'
        _ => b'g'
      }
  }
}

///|
fn luna_http1_content_type_byte(index : Int) -> Byte {
  match index {
    0 => b'a'
    1 => b'p'
    2 => b'p'
    3 => b'l'
    4 => b'i'
    5 => b'c'
    6 => b'a'
    7 => b't'
    8 => b'i'
    9 => b'o'
    10 => b'n'
    11 => b'/'
    12 => b'j'
    13 => b's'
    14 => b'o'
    _ => b'n'
  }
}

///|
fn luna_http1_bearer_prefix_byte(index : Int) -> Byte {
  match index {
    0 => b'b'
    1 => b'e'
    2 => b'a'
    3 => b'r'
    4 => b'e'
    5 => b'r'
    _ => b' '
  }
}

///|
fn LunaHttp1Workspace::progress_header_name_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.line_start == self.line_end {
    self.fail(LunaHttp1Header, LunaHttp1Invalid)
    return
  }
  let byte = self.storage[self.line_start]
  self.line_start += 1
  if byte == b':' {
    if self.header_name_index == 0 {
      self.fail(LunaHttp1Header, LunaHttp1Invalid)
      return
    }
    self.header_kind = if self.header_authorization &&
      self.header_name_index == 13 {
      LUNA_HTTP1_HEADER_AUTHORIZATION
    } else if self.header_content_length && self.header_name_index == 14 {
      LUNA_HTTP1_HEADER_CONTENT_LENGTH
    } else if self.header_content_type && self.header_name_index == 12 {
      LUNA_HTTP1_HEADER_CONTENT_TYPE
    } else if self.header_transfer_encoding && self.header_name_index == 17 {
      LUNA_HTTP1_HEADER_TRANSFER_ENCODING
    } else {
      LUNA_HTTP1_HEADER_UNKNOWN
    }
    if self.header_kind == LUNA_HTTP1_HEADER_TRANSFER_ENCODING {
      self.fail(LunaHttp1TransferEncoding, LunaHttp1Unsupported)
      return
    }
    if self.header_kind == LUNA_HTTP1_HEADER_AUTHORIZATION &&
      self.authorization_seen {
      self.fail(LunaHttp1Authorization, LunaHttp1Duplicate)
      return
    }
    if self.header_kind == LUNA_HTTP1_HEADER_CONTENT_TYPE &&
      self.content_type_seen {
      self.fail(LunaHttp1ContentType, LunaHttp1Duplicate)
      return
    }
    self.duplicate_content_length = self.header_kind ==
      LUNA_HTTP1_HEADER_CONTENT_LENGTH &&
      self.content_length_seen
    self.value_index = 0
    self.value_leading_space = false
    self.value_number = 0
    self.value_has_digit = false
    self.value_content_type = true
    if self.header_kind == LUNA_HTTP1_HEADER_AUTHORIZATION {
      self.credential_length = 0
      self.credential_padding = false
    }
    self.phase = LUNA_HTTP1_VALIDATING_HEADER_VALUE
    return
  }
  if !luna_http1_token_byte(byte) {
    self.fail(LunaHttp1Header, LunaHttp1Invalid)
    return
  }
  let lower = luna_http1_lower_ascii(byte)
  if self.header_name_index >= 13 ||
    lower !=
    luna_http1_header_name_byte(
      LUNA_HTTP1_HEADER_AUTHORIZATION,
      self.header_name_index,
    ) {
    self.header_authorization = false
  }
  if self.header_name_index >= 14 ||
    lower !=
    luna_http1_header_name_byte(
      LUNA_HTTP1_HEADER_CONTENT_LENGTH,
      self.header_name_index,
    ) {
    self.header_content_length = false
  }
  if self.header_name_index >= 12 ||
    lower !=
    luna_http1_header_name_byte(
      LUNA_HTTP1_HEADER_CONTENT_TYPE,
      self.header_name_index,
    ) {
    self.header_content_type = false
  }
  if self.header_name_index >= 17 ||
    lower !=
    luna_http1_header_name_byte(
      LUNA_HTTP1_HEADER_TRANSFER_ENCODING,
      self.header_name_index,
    ) {
    self.header_transfer_encoding = false
  }
  self.header_name_index += 1
}

///|
fn LunaHttp1Workspace::finish_header_value(self : LunaHttp1Workspace) -> Unit {
  match self.header_kind {
    LUNA_HTTP1_HEADER_AUTHORIZATION => {
      if self.value_index <= 7 || self.credential_length == 0 {
        self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
        return
      }
      self.authorization_seen = true
    }
    LUNA_HTTP1_HEADER_CONTENT_LENGTH => {
      if !self.value_has_digit {
        self.fail(LunaHttp1ContentLength, LunaHttp1Invalid)
        return
      }
      if self.duplicate_content_length {
        self.fail(
          LunaHttp1ContentLength,
          if self.value_number == self.body_length {
            LunaHttp1Duplicate
          } else {
            LunaHttp1Conflicting
          },
        )
        return
      }
      self.body_length = self.value_number
      self.content_length_seen = true
    }
    LUNA_HTTP1_HEADER_CONTENT_TYPE => {
      if self.value_index != 16 || !self.value_content_type {
        self.fail(LunaHttp1ContentType, LunaHttp1Unsupported)
        return
      }
      self.content_type_seen = true
    }
    _ => ()
  }
  self.line_start = self.scan_cursor
  self.phase = LUNA_HTTP1_SCANNING_LINE
}

///|
fn LunaHttp1Workspace::progress_header_value_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.line_start == self.line_end {
    self.finish_header_value()
    return
  }
  let byte = self.storage[self.line_start]
  let byte_offset = self.line_start
  self.line_start += 1
  if self.value_index == 0 && !self.value_leading_space && byte == b' ' {
    self.value_leading_space = true
    return
  }
  match self.header_kind {
    LUNA_HTTP1_HEADER_AUTHORIZATION => {
      if self.value_index < 7 {
        let expected = luna_http1_bearer_prefix_byte(self.value_index)
        if (self.value_index < 6 && luna_http1_lower_ascii(byte) != expected) ||
          (self.value_index == 6 && byte != expected) {
          self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
          return
        }
      } else {
        if !luna_http1_bearer_byte(byte, self.credential_padding) {
          self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
          return
        }
        if self.credential_length == 0 {
          self.credential_offset = byte_offset
        }
        if byte == b'=' {
          self.credential_padding = true
        }
        self.credential_length += 1
      }
      self.value_index += 1
    }
    LUNA_HTTP1_HEADER_CONTENT_LENGTH => {
      if byte < b'0' || byte > b'9' {
        self.fail(LunaHttp1ContentLength, LunaHttp1Invalid)
        return
      }
      let digit = byte.to_int() - b'0'.to_int()
      if self.value_number > (self.limits.max_body_bytes - digit) / 10 {
        self.fail(LunaHttp1ContentLength, LunaHttp1TooLong)
        return
      }
      self.value_number = self.value_number * 10 + digit
      self.value_has_digit = true
      self.value_index += 1
    }
    LUNA_HTTP1_HEADER_CONTENT_TYPE => {
      if self.value_index >= 16 ||
        luna_http1_lower_ascii(byte) !=
        luna_http1_content_type_byte(self.value_index) {
        self.value_content_type = false
      }
      self.value_index += 1
    }
    _ => self.value_index += 1
  }
}

///|
fn LunaHttp1Workspace::progress_required_one(self : LunaHttp1Workspace) -> Unit {
  match self.parse_index {
    0 =>
      if !self.authorization_seen {
        self.fail(LunaHttp1Authorization, LunaHttp1Missing)
        return
      }
    1 =>
      if !self.content_length_seen {
        self.fail(LunaHttp1ContentLength, LunaHttp1Missing)
        return
      }
    2 =>
      if !self.content_type_seen {
        self.fail(LunaHttp1ContentType, LunaHttp1Missing)
        return
      }
    _ => {
      self.phase = LUNA_HTTP1_AUTH_BEGIN
      return
    }
  }
  self.parse_index += 1
}

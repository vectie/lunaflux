///|
fn luna_http1_lower_ascii(byte : Byte) -> Byte {
  if byte >= b'A' && byte <= b'Z' {
    (byte.to_int() + 32).to_byte()
  } else {
    byte
  }
}

///|
fn luna_http1_token_byte(byte : Byte) -> Bool {
  (byte >= b'a' && byte <= b'z') ||
  (byte >= b'A' && byte <= b'Z') ||
  (byte >= b'0' && byte <= b'9') ||
  byte == b'-'
}

///|
fn luna_http1_bearer_byte(byte : Byte, padding : Bool) -> Bool {
  if byte == b'=' {
    true
  } else if padding {
    false
  } else {
    (byte >= b'a' && byte <= b'z') ||
    (byte >= b'A' && byte <= b'Z') ||
    (byte >= b'0' && byte <= b'9') ||
    byte == b'-' ||
    byte == b'.' ||
    byte == b'_' ||
    byte == b'~' ||
    byte == b'+' ||
    byte == b'/'
  }
}

///|
fn luna_http1_request_byte(index : Int) -> Byte {
  match index {
    0 => b'P'
    1 => b'O'
    2 => b'S'
    _ => b'T'
  }
}

///|
fn luna_http1_responses_byte(index : Int) -> Byte {
  match index {
    0 => b'/'
    1 => b'v'
    2 => b'1'
    3 => b'/'
    4 => b'r'
    5 => b'e'
    6 => b's'
    7 => b'p'
    8 => b'o'
    9 => b'n'
    10 => b's'
    11 => b'e'
    _ => b's'
  }
}

///|
fn luna_http1_chat_byte(index : Int) -> Byte {
  match index {
    0 => b'/'
    1 => b'v'
    2 => b'1'
    3 => b'/'
    4 => b'c'
    5 => b'h'
    6 => b'a'
    7 => b't'
    8 => b'/'
    9 => b'c'
    10 => b'o'
    11 => b'm'
    12 => b'p'
    13 => b'l'
    14 => b'e'
    15 => b't'
    16 => b'i'
    17 => b'o'
    18 => b'n'
    _ => b's'
  }
}

///|
fn luna_http1_version_byte(index : Int) -> Byte {
  match index {
    0 => b'H'
    1 => b'T'
    2 => b'T'
    3 => b'P'
    4 => b'/'
    5 => b'1'
    6 => b'.'
    _ => b'1'
  }
}

///|
fn LunaHttp1Workspace::begin_request_validation(
  self : LunaHttp1Workspace,
) -> Unit {
  self.phase = LUNA_HTTP1_VALIDATING_REQUEST
  self.parse_component = 0
  self.parse_index = 0
  self.route_responses = true
  self.route_chat = true
}

///|
fn LunaHttp1Workspace::begin_header_validation(
  self : LunaHttp1Workspace,
) -> Unit {
  self.phase = LUNA_HTTP1_VALIDATING_HEADER_NAME
  self.header_name_index = 0
  self.header_authorization = true
  self.header_content_length = true
  self.header_content_type = true
  self.header_transfer_encoding = true
}

///|
fn LunaHttp1Workspace::progress_scan_line_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.scan_cursor >= self.head_length {
    abort("Luna HTTP/1 complete-head scan invariant")
  }
  let byte = self.storage[self.scan_cursor]
  if self.scan_expect_lf {
    if byte != b'\n' {
      self.fail(LunaHttp1Header, LunaHttp1Invalid)
      return
    }
    self.scan_cursor += 1
    self.scan_expect_lf = false
    if self.line_end == self.line_start {
      if !self.request_line_complete {
        self.fail(LunaHttp1RequestLine, LunaHttp1Invalid)
      } else {
        self.phase = LUNA_HTTP1_VALIDATING_REQUIRED
        self.parse_index = 0
      }
    } else if self.request_line_complete {
      self.begin_header_validation()
    } else {
      self.begin_request_validation()
    }
    return
  }
  if self.scan_cursor == self.line_start &&
    self.request_line_complete &&
    byte == b' ' {
    self.fail(LunaHttp1Header, LunaHttp1Invalid)
    return
  }
  if byte == b'\r' {
    self.line_end = self.scan_cursor
    self.scan_cursor += 1
    self.scan_expect_lf = true
    return
  }
  if byte == b'\n' || byte < b' ' || byte >= b'\x7f' {
    self.fail(
      if self.request_line_complete {
        LunaHttp1Header
      } else {
        LunaHttp1RequestLine
      },
      LunaHttp1Invalid,
    )
    return
  }
  self.scan_cursor += 1
}

///|
fn LunaHttp1Workspace::progress_request_one(self : LunaHttp1Workspace) -> Unit {
  if self.line_start == self.line_end {
    match self.parse_component {
      0 => self.fail(LunaHttp1Method, LunaHttp1Unsupported)
      1 => self.fail(LunaHttp1RouteRule, LunaHttp1Unsupported)
      _ => {
        if self.parse_index != 8 || !self.route_responses {
          self.fail(LunaHttp1Version, LunaHttp1Unsupported)
          return
        }
        self.request_line_complete = true
        self.line_start = self.scan_cursor
        self.phase = LUNA_HTTP1_SCANNING_LINE
      }
    }
    return
  }
  let byte = self.storage[self.line_start]
  self.line_start += 1
  match self.parse_component {
    0 =>
      if byte == b' ' {
        if self.parse_index != 4 {
          self.fail(LunaHttp1Method, LunaHttp1Unsupported)
        } else {
          self.parse_component = 1
          self.parse_index = 0
          self.route_responses = true
          self.route_chat = true
        }
      } else {
        if self.parse_index >= 4 ||
          byte != luna_http1_request_byte(self.parse_index) {
          self.fail(LunaHttp1Method, LunaHttp1Unsupported)
          return
        }
        self.parse_index += 1
      }
    1 =>
      if byte == b' ' {
        if self.route_responses && self.parse_index == 13 {
          self.route_code = 1
        } else if self.route_chat && self.parse_index == 20 {
          self.route_code = 2
        } else {
          self.fail(LunaHttp1RouteRule, LunaHttp1Unsupported)
          return
        }
        self.parse_component = 2
        self.parse_index = 0
        self.route_responses = true
      } else {
        if self.parse_index >= 13 ||
          byte != luna_http1_responses_byte(self.parse_index) {
          self.route_responses = false
        }
        if self.parse_index >= 20 ||
          byte != luna_http1_chat_byte(self.parse_index) {
          self.route_chat = false
        }
        self.parse_index += 1
      }
    _ => {
      if self.parse_index >= 8 ||
        byte != luna_http1_version_byte(self.parse_index) {
        self.route_responses = false
      }
      self.parse_index += 1
    }
  }
}

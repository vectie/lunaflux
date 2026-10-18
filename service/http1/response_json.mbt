///|
fn json_head_prefix(failed : Bool) -> Bytes {
  if failed {
    b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
  } else {
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
  }
}

///|
pub fn LunaHttp1ResponseWorkspace::begin_json(
  self : Self,
  content_length~ : Int,
  failed? : Bool = false,
) -> LunaHttp1ResponseWork raise LunaHttp1Error {
  guard content_length >= 0 else {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1Invalid)
  }
  let digits = for value = content_length, count = 1 {
    if value < 10 {
      break count
    }
    continue value / 10, count + 1
  }
  let work = self.begin_kind(
    if failed {
      6
    } else {
      5
    },
    if failed {
      500
    } else {
      200
    },
    json_head_prefix(failed).length() + digits + 4,
  )
  self.body_length = content_length
  work
}

///|
fn LunaHttp1ResponseWorkspace::json_head_byte(self : Self, index : Int) -> Byte {
  let prefix = json_head_prefix(self.response_kind == 6)
  if index < prefix.length() {
    return prefix[index]
  }
  let digits = self.response_length - prefix.length() - 4
  let at = index - prefix.length()
  if at >= digits {
    let suffix = b"\r\n\r\n"
    return suffix[at - digits]
  }
  let divisor = for index = 0, value = 1 {
    if index == digits - at - 1 {
      break value
    }
    continue index + 1, value * 10
  }
  (48 + self.body_length / divisor % 10).to_byte()
}

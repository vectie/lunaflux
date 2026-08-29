///|
fn luna_http1_response_http_prefix(index : Int) -> Byte {
  match index {
    0 => b'H'
    1 => b'T'
    2 => b'T'
    3 => b'P'
    4 => b'/'
    5 => b'1'
    6 => b'.'
    7 => b'1'
    _ => b' '
  }
}

///|
fn luna_http1_response_status_byte(kind : Int, index : Int) -> Byte {
  match kind {
    0 =>
      match index {
        0 => b'2'
        1 => b'0'
        2 => b'0'
        3 => b' '
        4 => b'O'
        5 => b'K'
        6 => b'\r'
        _ => b'\n'
      }
    1 =>
      match index {
        0 => b'4'
        1 => b'0'
        2 => b'1'
        3 => b' '
        4 => b'U'
        5 => b'n'
        6 => b'a'
        7 => b'u'
        8 => b't'
        9 => b'h'
        10 => b'o'
        11 => b'r'
        12 => b'i'
        13 => b'z'
        14 => b'e'
        15 => b'd'
        16 => b'\r'
        _ => b'\n'
      }
    2 =>
      match index {
        0 => b'4'
        1 => b'0'
        2 => b'0'
        3 => b' '
        4 => b'B'
        5 => b'a'
        6 => b'd'
        7 => b' '
        8 => b'R'
        9 => b'e'
        10 => b'q'
        11 => b'u'
        12 => b'e'
        13 => b's'
        14 => b't'
        15 => b'\r'
        _ => b'\n'
      }
    3 =>
      match index {
        0 => b'4'
        1 => b'2'
        2 => b'9'
        3 => b' '
        4 => b'T'
        5 => b'o'
        6 => b'o'
        7 => b' '
        8 => b'M'
        9 => b'a'
        10 => b'n'
        11 => b'y'
        12 => b' '
        13 => b'R'
        14 => b'e'
        15 => b'q'
        16 => b'u'
        17 => b'e'
        18 => b's'
        19 => b't'
        20 => b's'
        21 => b'\r'
        _ => b'\n'
      }
    _ =>
      match index {
        0 => b'5'
        1 => b'0'
        2 => b'3'
        3 => b' '
        4 => b'S'
        5 => b'e'
        6 => b'r'
        7 => b'v'
        8 => b'i'
        9 => b'c'
        10 => b'e'
        11 => b' '
        12 => b'U'
        13 => b'n'
        14 => b'a'
        15 => b'v'
        16 => b'a'
        17 => b'i'
        18 => b'l'
        19 => b'a'
        20 => b'b'
        21 => b'l'
        22 => b'e'
        23 => b'\r'
        _ => b'\n'
      }
  }
}

///|
fn luna_http1_response_content_prefix(index : Int) -> Byte {
  match index {
    0 => b'C'
    1 => b'o'
    2 => b'n'
    3 => b't'
    4 => b'e'
    5 => b'n'
    6 => b't'
    7 => b'-'
    8 => b'T'
    9 => b'y'
    10 => b'p'
    11 => b'e'
    12 => b':'
    _ => b' '
  }
}

///|
fn luna_http1_response_event_type(index : Int) -> Byte {
  match index {
    0 => b't'
    1 => b'e'
    2 => b'x'
    3 => b't'
    4 => b'/'
    5 => b'e'
    6 => b'v'
    7 => b'e'
    8 => b'n'
    9 => b't'
    10 => b'-'
    11 => b's'
    12 => b't'
    13 => b'r'
    14 => b'e'
    15 => b'a'
    _ => b'm'
  }
}

///|
fn luna_http1_response_json_type(index : Int) -> Byte {
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
fn luna_http1_response_cache_byte(index : Int) -> Byte {
  match index {
    0 => b'C'
    1 => b'a'
    2 => b'c'
    3 => b'h'
    4 => b'e'
    5 => b'-'
    6 => b'C'
    7 => b'o'
    8 => b'n'
    9 => b't'
    10 => b'r'
    11 => b'o'
    12 => b'l'
    13 => b':'
    14 => b' '
    15 => b'n'
    16 => b'o'
    17 => b'-'
    18 => b'c'
    19 => b'a'
    20 => b'c'
    21 => b'h'
    22 => b'e'
    23 => b'\r'
    _ => b'\n'
  }
}

///|
fn luna_http1_response_connection_byte(index : Int) -> Byte {
  match index {
    0 => b'C'
    1 => b'o'
    2 => b'n'
    3 => b'n'
    4 => b'e'
    5 => b'c'
    6 => b't'
    7 => b'i'
    8 => b'o'
    9 => b'n'
    10 => b':'
    11 => b' '
    12 => b'c'
    13 => b'l'
    14 => b'o'
    15 => b's'
    16 => b'e'
    17 => b'\r'
    18 => b'\n'
    19 => b'\r'
    _ => b'\n'
  }
}

///|
fn luna_http1_response_length_prefix(index : Int) -> Byte {
  match index {
    0 => b'C'
    1 => b'o'
    2 => b'n'
    3 => b't'
    4 => b'e'
    5 => b'n'
    6 => b't'
    7 => b'-'
    8 => b'L'
    9 => b'e'
    10 => b'n'
    11 => b'g'
    12 => b't'
    13 => b'h'
    14 => b':'
    _ => b' '
  }
}

///|
fn luna_http1_response_error_prefix(index : Int) -> Byte {
  match index {
    0 => b'{'
    1 => b'"'
    2 => b'e'
    3 => b'r'
    4 => b'r'
    5 => b'o'
    6 => b'r'
    7 => b'"'
    8 => b':'
    9 => b'{'
    10 => b'"'
    11 => b't'
    12 => b'y'
    13 => b'p'
    14 => b'e'
    15 => b'"'
    16 => b':'
    _ => b'"'
  }
}

///|
fn luna_http1_response_error_type_byte(kind : Int, index : Int) -> Byte {
  match kind {
    1 =>
      match index {
        0 => b'a'
        1 => b'u'
        2 => b't'
        3 => b'h'
        4 => b'e'
        5 => b'n'
        6 => b't'
        7 => b'i'
        8 => b'c'
        9 => b'a'
        10 => b't'
        11 => b'i'
        12 => b'o'
        13 => b'n'
        14 => b'_'
        15 => b'e'
        16 => b'r'
        17 => b'r'
        18 => b'o'
        _ => b'r'
      }
    2 =>
      match index {
        0 => b'v'
        1 => b'a'
        2 => b'l'
        3 => b'i'
        4 => b'd'
        5 => b'a'
        6 => b't'
        7 => b'i'
        8 => b'o'
        9 => b'n'
        10 => b'_'
        11 => b'e'
        12 => b'r'
        13 => b'r'
        14 => b'o'
        _ => b'r'
      }
    3 =>
      match index {
        0 => b'c'
        1 => b'a'
        2 => b'p'
        3 => b'a'
        4 => b'c'
        5 => b'i'
        6 => b't'
        7 => b'y'
        8 => b'_'
        9 => b'e'
        10 => b'r'
        11 => b'r'
        12 => b'o'
        _ => b'r'
      }
    _ =>
      match index {
        0 => b's'
        1 => b'e'
        2 => b'r'
        3 => b'v'
        4 => b'i'
        5 => b'c'
        6 => b'e'
        7 => b'_'
        8 => b'e'
        9 => b'r'
        10 => b'r'
        11 => b'o'
        _ => b'r'
      }
  }
}

///|
fn luna_http1_response_status_length(kind : Int) -> Int {
  match kind {
    0 => 8
    1 => 18
    2 => 17
    3 => 23
    _ => 25
  }
}

///|
fn luna_http1_response_error_type_length(kind : Int) -> Int {
  match kind {
    1 => 20
    2 => 16
    3 => 14
    _ => 13
  }
}

///|
fn luna_http1_response_content_length(kind : Int) -> Int {
  match kind {
    1 => 41
    2 => 37
    3 => 35
    _ => 34
  }
}

///|
fn luna_http1_response_byte(kind : Int, index : Int) -> Byte {
  if index < 9 {
    return luna_http1_response_http_prefix(index)
  }
  let status_length = luna_http1_response_status_length(kind)
  let mut cursor = index - 9
  if cursor < status_length {
    return luna_http1_response_status_byte(kind, cursor)
  }
  cursor -= status_length
  if cursor < 14 {
    return luna_http1_response_content_prefix(cursor)
  }
  cursor -= 14
  if kind == 0 {
    if cursor < 17 {
      return luna_http1_response_event_type(cursor)
    }
    cursor -= 17
    if cursor < 2 {
      return if cursor == 0 { b'\r' } else { b'\n' }
    }
    cursor -= 2
    if cursor < 25 {
      return luna_http1_response_cache_byte(cursor)
    }
    return luna_http1_response_connection_byte(cursor - 25)
  }
  if cursor < 16 {
    return luna_http1_response_json_type(cursor)
  }
  cursor -= 16
  if cursor < 2 {
    return if cursor == 0 { b'\r' } else { b'\n' }
  }
  cursor -= 2
  if kind == 1 {
    if cursor < 26 {
      return luna_http1_response_challenge_byte(cursor)
    }
    cursor -= 26
  }
  if cursor < 16 {
    return luna_http1_response_length_prefix(cursor)
  }
  cursor -= 16
  let content_length = luna_http1_response_content_length(kind)
  if cursor == 0 {
    return (b'0'.to_int() + content_length / 10).to_byte()
  }
  if cursor == 1 {
    return (b'0'.to_int() + content_length % 10).to_byte()
  }
  if cursor == 2 {
    return b'\r'
  }
  if cursor == 3 {
    return b'\n'
  }
  cursor -= 4
  if cursor < 21 {
    return luna_http1_response_connection_byte(cursor)
  }
  cursor -= 21
  if cursor < 18 {
    return luna_http1_response_error_prefix(cursor)
  }
  cursor -= 18
  let type_length = luna_http1_response_error_type_length(kind)
  if cursor < type_length {
    return luna_http1_response_error_type_byte(kind, cursor)
  }
  match cursor - type_length {
    0 => b'"'
    1 => b'}'
    _ => b'}'
  }
}

///|

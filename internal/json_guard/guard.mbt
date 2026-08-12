///|
/// Bounded, payload-safe failures from lexical JSON structure validation.
pub(all) suberror JsonGuardError {
  InvalidDepthLimit
  DepthLimitExceeded(limit~ : Int)
  DuplicateObjectKey(depth~ : Int, index~ : Int)
  MalformedStructure
} derive(Debug, Eq)

///|
/// Rejects duplicate JSON object keys before a map-backed JSON parser can
/// overwrite them.
///
/// Keys are decoded before comparison, so escaped-equivalent spellings are
/// duplicates. Object and array nesting is bounded by `max_depth`. The input
/// itself must already have passed its caller-owned byte bound; this guard's
/// work and allocation are then bounded by that input and the depth limit.
pub fn reject_duplicate_object_keys(
  input : StringView,
  max_depth~ : Int,
) -> Unit raise JsonGuardError {
  if max_depth <= 0 {
    raise InvalidDepthLimit
  }
  let scanner = Scanner::new(input, max_depth)
  scanner.skip_whitespace()
  scanner.scan_value(0)
  scanner.skip_whitespace()
  if !scanner.at_end() {
    raise MalformedStructure
  }
}

///|
priv struct Scanner {
  input : StringView
  max_depth : Int
  mut position : Int
}

///|
fn Scanner::new(input : StringView, max_depth : Int) -> Scanner {
  { input, max_depth, position: 0 }
}

///|
fn Scanner::at_end(self : Scanner) -> Bool {
  self.position >= self.input.length()
}

///|
fn Scanner::peek(self : Scanner) -> UInt16? {
  if self.at_end() {
    None
  } else {
    Some(self.input[self.position])
  }
}

///|
fn Scanner::advance(self : Scanner) -> UInt16 raise JsonGuardError {
  match self.peek() {
    Some(unit) => {
      self.position += 1
      unit
    }
    None => raise MalformedStructure
  }
}

///|
fn Scanner::skip_whitespace(self : Scanner) -> Unit {
  while self.peek() is Some(' ' | '\n' | '\r' | '\t') {
    self.position += 1
  }
}

///|
fn Scanner::scan_value(
  self : Scanner,
  depth : Int,
) -> Unit raise JsonGuardError {
  self.skip_whitespace()
  match self.peek() {
    Some('{') => self.scan_object(depth + 1)
    Some('[') => self.scan_array(depth + 1)
    Some('"') => ignore(self.scan_string())
    Some('t') => self.scan_literal("true")
    Some('f') => self.scan_literal("false")
    Some('n') => self.scan_literal("null")
    Some('-' | '0'..='9') => self.scan_number()
    _ => raise MalformedStructure
  }
}

///|
fn Scanner::check_depth(
  self : Scanner,
  depth : Int,
) -> Unit raise JsonGuardError {
  if depth > self.max_depth {
    raise DepthLimitExceeded(limit=self.max_depth)
  }
}

///|
fn Scanner::scan_object(
  self : Scanner,
  depth : Int,
) -> Unit raise JsonGuardError {
  self.check_depth(depth)
  ignore(self.advance())
  self.skip_whitespace()
  if self.peek() is Some('}') {
    ignore(self.advance())
    return
  }
  let keys : Map[String, Unit] = Map([])
  let mut index = 0
  for ;; {
    if !(self.peek() is Some('"')) {
      raise MalformedStructure
    }
    let key = self.scan_string()
    if keys.contains(key) {
      raise DuplicateObjectKey(depth~, index~)
    }
    keys[key] = ()
    self.skip_whitespace()
    if self.advance() != ':' {
      raise MalformedStructure
    }
    self.scan_value(depth)
    self.skip_whitespace()
    match self.advance() {
      '}' => return
      ',' => {
        index += 1
        self.skip_whitespace()
      }
      _ => raise MalformedStructure
    }
  }
}

///|
fn Scanner::scan_array(
  self : Scanner,
  depth : Int,
) -> Unit raise JsonGuardError {
  self.check_depth(depth)
  ignore(self.advance())
  self.skip_whitespace()
  if self.peek() is Some(']') {
    ignore(self.advance())
    return
  }
  for ;; {
    self.scan_value(depth)
    self.skip_whitespace()
    match self.advance() {
      ']' => return
      ',' => self.skip_whitespace()
      _ => raise MalformedStructure
    }
  }
}

///|
fn Scanner::scan_literal(
  self : Scanner,
  expected : StringView,
) -> Unit raise JsonGuardError {
  for index in 0..<expected.length() {
    if self.advance() != expected[index] {
      raise MalformedStructure
    }
  }
}

///|
fn Scanner::scan_number(self : Scanner) -> Unit raise JsonGuardError {
  let start = self.position
  while self.peek() is Some('-' | '+' | '.' | '0'..='9' | 'e' | 'E') {
    self.position += 1
  }
  if self.position == start {
    raise MalformedStructure
  }
}

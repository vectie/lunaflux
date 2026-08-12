///|
fn Scanner::scan_string(self : Scanner) -> String raise JsonGuardError {
  if self.advance() != '"' {
    raise MalformedStructure
  }
  let output = StringBuilder()
  for ;; {
    let unit = self.advance()
    match unit {
      '"' => return output.to_string()
      '\\' => self.scan_escape(output)
      '\u{0000}'..='\u{001f}' => raise MalformedStructure
      _ => {
        let character = self.input.get_char(self.position - 1)
        match character {
          Some(char) => {
            output.write_char(char)
            self.position += char.utf16_len() - 1
          }
          None => raise MalformedStructure
        }
      }
    }
  }
}

///|
fn Scanner::scan_escape(
  self : Scanner,
  output : StringBuilder,
) -> Unit raise JsonGuardError {
  match self.advance() {
    '"' => output.write_char('"')
    '\\' => output.write_char('\\')
    '/' => output.write_char('/')
    'b' => output.write_char('\b')
    'f' => output.write_char('\f')
    'n' => output.write_char('\n')
    'r' => output.write_char('\r')
    't' => output.write_char('\t')
    'u' => self.scan_unicode_escape(output)
    _ => raise MalformedStructure
  }
}

///|
fn Scanner::scan_unicode_escape(
  self : Scanner,
  output : StringBuilder,
) -> Unit raise JsonGuardError {
  let first = self.scan_hex_quad()
  let scalar = if first >= 0xd800 && first <= 0xdbff {
    if self.advance() != '\\' || self.advance() != 'u' {
      raise MalformedStructure
    }
    let second = self.scan_hex_quad()
    if second < 0xdc00 || second > 0xdfff {
      raise MalformedStructure
    }
    0x10000 + (first - 0xd800) * 0x400 + (second - 0xdc00)
  } else if first >= 0xdc00 && first <= 0xdfff {
    raise MalformedStructure
  } else {
    first
  }
  match scalar.to_char() {
    Some(character) => output.write_char(character)
    None => raise MalformedStructure
  }
}

///|
fn Scanner::scan_hex_quad(self : Scanner) -> Int raise JsonGuardError {
  let mut value = 0
  for _ in 0..<4 {
    let digit = match self.advance() {
      '0'..='9' as unit => unit.to_int() - '0'
      'a'..='f' as unit => unit.to_int() - 'a' + 10
      'A'..='F' as unit => unit.to_int() - 'A' + 10
      _ => raise MalformedStructure
    }
    value = value * 16 + digit
  }
  value
}

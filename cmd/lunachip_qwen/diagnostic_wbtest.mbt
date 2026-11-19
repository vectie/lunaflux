///|
test "diagnostic greedy has stable ties and rejects nonfinite vocabulary" {
  assert_true(terminal_token(151645) && terminal_token(151643))
  assert_false(terminal_token(151644) || terminal_token(0))
  let values = Array::make(151936, -1.0F)
  values[42] = 2.0
  values[9000] = 2.0
  assert_eq(greedy(values[:]), 42)
  for bad in [0x7fc00000U, 0x7f800000U, 0xff800000U] {
    values[800] = Float::reinterpret_from_uint(bad)
    let mut rejected = false
    ignore(greedy(values[:])) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}

///|
test "wire float conversion preserves signed zeros subnormals and NaN bits" {
  let words = [0x80000000U, 1U, 0x7fc12345U, 0x3f800001U]
  let bytes = float_bytes(words.map(Float::reinterpret_from_uint)[:])
  assert_eq(decode_hex(@crypto.bytes_to_hex_string(bytes)[:]), bytes)
  for i, expected in words {
    assert_eq(decode_word(bytes, i), expected)
  }
  for invalid in ["0", "+1", "gg", "00 0"] {
    let mut rejected = false
    ignore(decode_hex(invalid[:])) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}

///|
async test "stale or malformed JTAG replies poison ownership without a retry" {
  for
    response in ["0 OK 01000000\n", "1 OK 0100\n", "1 OK gg000000\n", "ERROR\n"] {
    let server = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
    defer server.close()
    @async.with_timeout(3000, () => {
      @async.with_task_group(group => {
        group.spawn_bg(() => {
          let connection = server.accept().0
          defer connection.close()
          assert_eq(connection.read_until("\n"), Some("1 R 2148532224 4"))
          connection.write(response)
          assert_true(connection.read_some() is None)
        })
        let connection = @socket.Tcp::connect(server.addr)
        defer connection.close()
        let jtag = Jtag::{
          connection,
          sequence: 0,
          owner: @device.LaunchOwner::new(),
        }
        let mut failed = false
        ignore(jtag.word(0x80100000UL)) catch {
          _ => failed = true
        }
        assert_true(failed)
        try jtag.owner.begin(0U, 1) catch {
          @device.InvalidState => ()
          _ => fail("wrong poison error")
        } noraise {
          _ => fail("uncertain transport owner reused")
        }
      })
    })
  }
}

///|
async test "JTAG host preserves payload byte order and increasing request identity" {
  let server = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
  defer server.close()
  @async.with_timeout(3000, () => {
    @async.with_task_group(group => {
      group.spawn_bg(() => {
        let connection = server.accept().0
        defer connection.close()
        assert_eq(connection.read_until("\n"), Some("1 W 64 78563412"))
        connection.write("1 OK \n")
        assert_eq(connection.read_until("\n"), Some("2 R 64 4"))
        connection.write("2 OK 78563412\n")
      })
      let connection = @socket.Tcp::connect(server.addr)
      defer connection.close()
      let jtag = Jtag::{
        connection,
        sequence: 0,
        owner: @device.LaunchOwner::new(),
      }
      jtag.set_word(64UL, 0x12345678U)
      assert_eq(jtag.word(64UL), 0x12345678U)
      assert_eq(jtag.sequence, 2)
    })
  })
}

///|
async test "full vocabulary JTAG read survives fragmented megabyte response" {
  let payload = Bytes::from_array(
    Array::makei(151936 * 4, i => ((i * 37 + i / 251 + 19) % 256).to_byte())[:],
  )
  let server = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
  defer server.close()
  @async.with_timeout(10000, () => {
    @async.with_task_group(group => {
      group.spawn_bg(() => {
        let connection = server.accept().0
        defer connection.close()
        assert_eq(connection.read_until("\n"), Some("1 R 4096 607744"))
        // The dependency's generic hex helper prepends immutable strings and
        // is quadratic at this size. Build the independent mock line linearly.
        let encoded = StringBuilder()
        encoded.write_string("1 OK ")
        for byte in payload {
          for digit in [byte.to_int() >> 4, byte.to_int() & 15] {
            encoded.write_char(
              (if digit < 10 { 48 + digit } else { 87 + digit }).unsafe_to_char(),
            )
          }
        }
        encoded.write_char('\n')
        let response = encoded.to_string()
        let mut first = 0
        while first < response.length() {
          // Odd fragments split encoded bytes and cross socket buffer sizes.
          let end = (first + 8191).min(response.length())
          connection.write(response[first:end].to_owned())
          first = end
        }
      })
      let connection = @socket.Tcp::connect(server.addr)
      defer connection.close()
      let jtag = Jtag::{
        connection,
        sequence: 0,
        owner: @device.LaunchOwner::new(),
      }
      assert_eq(jtag.read(4096UL, 607744), payload)
      assert_eq(jtag.sequence, 1)
    })
  })
}

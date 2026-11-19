///|
test "resident counters retain high32 bits and stage time separately" {
  let words = [
    0xffffffffU, 5U, 0U, 2048U, 2047U, 14U, 0U, 0x12345678U, 2U, 2047U, 99U, 100U,
    901U, 0xffffffffU,
  ]
  let snapshot = cycle_snapshot(words_bytes(words[:]))
  assert_eq(snapshot.core_cycles, 0x212345678UL)
  assert_eq(snapshot.loader_cycles, 901UL)
  assert_eq(snapshot.count, 2048U)
  assert_eq(snapshot.sequence, 0xffffffffU)
  assert_eq(snapshot.after, snapshot.sequence)
  assert_eq(snapshot.selected_token, None)
  let packet = [..words, 1U, 3U, 151935U]
  assert_eq(
    cycle_snapshot(words_bytes(packet[:]), chain=true, greedy=true).selected_token,
    Some(151935U),
  )
  packet[16] = 151936U
  try
    ignore(cycle_snapshot(words_bytes(packet[:]), chain=true, greedy=true))
  catch {
    _ => ()
  } noraise {
    _ => fail("out-of-range greedy result")
  }
  let payload = Bytes::from_array(
    Array::makei(65536, i => (i * 37).to_byte())[:],
  )
  assert_eq(decode_hex(encode_hex(payload[:])[:]), payload)
}

///|
async test "resident launch validates generation and counters without program upload" {
  for valid in [true, false] {
    let server = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
    defer server.close()
    @async.with_timeout(3000, () => {
      @async.with_task_group(group => {
        group.spawn_bg(() => {
          let c = server.accept().0
          defer c.close()
          assert_eq(c.read_until("\n"), Some("1 X 1610612736 549 4294967295"))
          let words = [
            0U,
            5U,
            0U,
            549U,
            548U,
            14U,
            0U,
            23U,
            1U,
            548U,
            4096U,
            512U,
            901U,
            if valid {
              0U
            } else {
              1U
            },
          ]
          c.write("1 OK \{encode_hex(words_bytes(words[:])[:])}\n")
        })
        let connection = @socket.Tcp::connect(server.addr)
        defer connection.close()
        let jtag = Jtag::{
          connection,
          sequence: 0,
          owner: @device.LaunchOwner::new(),
        }
        let mut rejected = false
        let result = try
          jtag.resident_execute(
            { address: 0x60000000UL, count: 549, },
            0xffffffffU,
          )
        catch {
          _ => {
            rejected = true
            None
          }
        } noraise {
          value => Some(value)
        }
        assert_eq(rejected, !valid)
        if result is Some(snapshot) {
          assert_eq(snapshot.core_cycles, 0x100000017UL)
        }
        if !valid {
          try ignore(jtag.owner.begin(0U, 1)) catch {
            _ => ()
          } noraise {
            _ => fail("invalid snapshot did not poison owner")
          }
        }
      })
    })
  }
}

///|
test "resident chain checks every boundary and exact aperture end" {
  let first = ResidentProgram::{ address: 0x7ffdffc0UL, count: 2048, }
  let last = ResidentProgram::{ address: 0x7fffffc0UL, count: 1, }
  assert_eq(packed_chain([first, last]).count, 2049)
  for
    xs in [
      [],
      [last, first],
      [first, first],
      [{ ..last, count: 2, }],
      [{ ..last, count: 0, }],
    ] {
    try ignore(packed_chain(xs)) catch {
      _ => ()
    } noraise {
      _ => fail("invalid chain")
    }
  }
}

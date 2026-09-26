///|
fn receive(
  owner : @rank_child_control.RankChildControl,
) -> @rank_child_control.RankChildInbound raise {
  owner.begin_receive()
  for ;; {
    if owner.progress_receive().has_inbound() {
      return owner.inbound()
    }
    ignore(owner.wait_receive(maximum_wait_millis=10))
  }
}

///|
fn send(owner : @rank_child_control.RankChildControl) -> Unit raise {
  while !owner.progress_send().is_complete() {
    ignore(owner.wait_send(maximum_wait_millis=10))
  }
}

///|
// Fixture-only rank: deterministic delayed transfers, no device resources.
fn main raise {
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=38912,
  )
  let owner = @rank_child_control.open_rank_child_control(
    limits,
    io_timeout_millis=10000,
    shutdown_timeout_millis=10000,
  )
  let configure = receive(owner)
  let payload = FixedArray::make(38912, b'\x00')
  configure.copy_payload_to(payload, destination_offset=0)
  let contract = @tensor_parallel_rank_configure.decode_rank_configure(
    payload,
    configure.payload_length(),
    configure.binding(),
  )
  let capability = contract.startup().streaming().unwrap()
  let mut control = @streaming_ir.Control::new(
    contract.startup().streaming_epoch(),
  )
  let mut original : @streaming_ir.Opcode = Spill
  let mut polls = 0
  let mut generation = 1
  let rank = contract.binding().rank()
  owner.begin_ready(configure, contract.binding(), previous_sequence_value=0UL)
  send(owner)
  for ;; {
    let inbound = receive(owner)
    match inbound.kind() {
      StreamingRequest => {
        inbound.copy_payload_to(payload, destination_offset=0)
        let request = @streaming_ir.decode(payload, inbound.payload_length())
        let command = request.command
        let decision = control.decide(command, capability)
        let status = if decision.action is Reject {
          @streaming_ir.Rejected
        } else {
          control = decision.next
          match command.opcode {
            Spill | Restore => {
              original = command.opcode
              polls = 0
              Pending
            }
            Evict => {
              generation += 1
              Evicted
            }
            Cancel => {
              control = control.completed()
              Cancelled
            }
            Poll => {
              polls += 1
              if polls < 1 + rank * 2 {
                Pending
              } else {
                control = control.completed()
                if original is Spill {
                  HostReady
                } else {
                  DeviceReady
                }
              }
            }
          }
        }
        let host : @streaming_ir.HostPage = if status is HostReady {
          { slot: 0, generation, }
        } else {
          command.host
        }
        @streaming_ir.encode({ command, status, host, }, payload)
        owner.begin_streaming_reply(inbound, payload)
        send(owner)
      }
      Drain => {
        owner.begin_drained(inbound)
        send(owner)
      }
      Close => {
        owner.begin_closed(inbound)
        send(owner)
        return
      }
      _ => fail("unexpected rank fixture command")
    }
  }
}

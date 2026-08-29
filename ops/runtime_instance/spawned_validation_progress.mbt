///|
fn spawned_validation_observation_allowed(
  kind : @online_session.LunaOnlineFramedObservationKind,
) -> Bool {
  match kind {
    LunaOnlineFramedAdmissionObservation
    | LunaOnlineFramedTokenObservation
    | LunaOnlineFramedUsageObservation
    | LunaOnlineFramedCompletionObservation => true
    LunaOnlineFramedCancellationObservation
    | LunaOnlineFramedDeadlineObservation
    | LunaOnlineFramedRequestFailureObservation
    | LunaOnlineFramedWorkerFailureObservation
    | LunaOnlineFramedWorkerRestartObservation
    | LunaOnlineFramedWorkerRequestTerminalObservation => false
  }
}

///|
fn offer_spawned_validation_frame(
  stream : @online_session.LunaOnlineFramedStream,
  frame : FixedArray[Byte],
  turn_limit : Int,
) -> Unit raise RuntimeSpawnedValidationError {
  let mut cursor = 0
  let mut turns = 0
  while cursor < frame.length() && turns < turn_limit {
    let ingress = stream.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    ) catch {
      _ => raise spawned_validation_error(SpawnedValidationIngress)
    }
    match ingress.kind() {
      LunaOnlineFramedIngressAccepted => {
        let consumed = ingress.consumed_bytes()
        if consumed <= 0 || consumed > frame.length() - cursor {
          raise spawned_validation_error(SpawnedValidationIngress)
        }
        cursor += consumed
      }
      LunaOnlineFramedIngressBackpressured => {
        let progress = stream.progress() catch {
          _ => raise spawned_validation_error(SpawnedValidationIngress)
        }
        match progress {
          LunaOnlineFramedCoordinatorRejectionReady => {
            let rejection = stream.take_rejection() catch {
              _ => raise spawned_validation_error(SpawnedValidationRejected)
            }
            rejection.ack() catch {
              _ => ()
            }
            raise spawned_validation_error(SpawnedValidationRejected)
          }
          LunaOnlineFramedCoordinatorClosed
          | LunaOnlineFramedCoordinatorDraining
          | LunaOnlineFramedCoordinatorEventReady
          | LunaOnlineFramedCoordinatorSemanticEventReady =>
            raise spawned_validation_error(SpawnedValidationIngress)
          _ => ()
        }
      }
      LunaOnlineFramedIngressDraining =>
        raise spawned_validation_error(SpawnedValidationIngress)
    }
    turns += 1
  }
  if cursor != frame.length() {
    raise spawned_validation_error(SpawnedValidationIngress)
  }
}

///|
fn next_spawned_validation_event(
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> @online_session.LunaOnlineFramedSemanticEvent raise RuntimeSpawnedValidationError {
  for _ in 0..<turn_limit {
    let progress = stream.progress() catch {
      _ => raise spawned_validation_error(SpawnedValidationEvent)
    }
    match progress {
      LunaOnlineFramedCoordinatorSemanticEventReady =>
        return stream.take_semantic_event() catch {
          _ => raise spawned_validation_error(SpawnedValidationEvent)
        }
      LunaOnlineFramedCoordinatorObservationReady => {
        let observation = stream.take_observation() catch {
          _ => raise spawned_validation_error(SpawnedValidationObservation)
        }
        let allowed = spawned_validation_observation_allowed(
          observation.kind() catch {
            _ => {
              observation.ack() catch {
                _ => ()
              }
              raise spawned_validation_error(SpawnedValidationObservation)
            }
          },
        )
        observation.ack() catch {
          _ => raise spawned_validation_error(SpawnedValidationObservation)
        }
        if !allowed {
          raise spawned_validation_error(SpawnedValidationObservation)
        }
      }
      LunaOnlineFramedCoordinatorRejectionReady => {
        let rejection = stream.take_rejection() catch {
          _ => raise spawned_validation_error(SpawnedValidationRejected)
        }
        rejection.ack() catch {
          _ => ()
        }
        raise spawned_validation_error(SpawnedValidationRejected)
      }
      LunaOnlineFramedCoordinatorClosed
      | LunaOnlineFramedCoordinatorDraining
      | LunaOnlineFramedCoordinatorEventReady
      | LunaOnlineFramedCoordinatorAwaitingInput =>
        raise spawned_validation_error(SpawnedValidationEvent)
      LunaOnlineFramedCoordinatorIdle
      | LunaOnlineFramedCoordinatorAdvanced
      | LunaOnlineFramedCoordinatorEventPreparing
      | LunaOnlineFramedCoordinatorMaintenanceRequired => ()
    }
  }
  raise spawned_validation_error(SpawnedValidationEvent)
}

///|
fn deliver_spawned_validation_event(
  event : @online_session.LunaOnlineFramedSemanticEvent,
) -> @luna_event.LunaEventView raise RuntimeSpawnedValidationError {
  event.view() catch {
    _ => raise spawned_validation_error(SpawnedValidationEvent)
  }
}

///|
fn finish_spawned_validation_event(
  event : @online_session.LunaOnlineFramedSemanticEvent,
) -> Unit raise RuntimeSpawnedValidationError {
  event.delivered() catch {
    _ => raise spawned_validation_error(SpawnedValidationEvent)
  }
}

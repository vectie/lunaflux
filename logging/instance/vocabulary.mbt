///|
fn luna_instance_log_event_code(event : LunaInstanceLogEvent) -> Int {
  match event {
    LunaInstanceStarting => 0
    LunaInstanceReady => 1
    LunaAdmissionAccepted => 2
    LunaAdmissionRejected => 3
    LunaRequestCompleted => 4
    LunaRequestCancelled => 5
    LunaRequestDeadline => 6
    LunaRequestFailed => 7
    LunaWorkerRestarted => 8
    LunaWorkerFailed => 9
    LunaNetworkAccepted => 10
    LunaNetworkDisconnected => 11
    LunaNetworkRejected => 12
    LunaBackpressureObserved => 13
    LunaDrainStarted => 14
    LunaInstanceClosed => 15
  }
}

///|
fn luna_instance_log_event_from_code(code : Int) -> LunaInstanceLogEvent {
  match code {
    0 => LunaInstanceStarting
    1 => LunaInstanceReady
    2 => LunaAdmissionAccepted
    3 => LunaAdmissionRejected
    4 => LunaRequestCompleted
    5 => LunaRequestCancelled
    6 => LunaRequestDeadline
    7 => LunaRequestFailed
    8 => LunaWorkerRestarted
    9 => LunaWorkerFailed
    10 => LunaNetworkAccepted
    11 => LunaNetworkDisconnected
    12 => LunaNetworkRejected
    13 => LunaBackpressureObserved
    14 => LunaDrainStarted
    _ => LunaInstanceClosed
  }
}

///|
fn luna_instance_log_reason_code(reason : LunaInstanceLogReason) -> Int {
  match reason {
    LunaLogNoReason => 0
    LunaLogAuthentication => 1
    LunaLogValidation => 2
    LunaLogCapacity => 3
    LunaLogDeadline => 4
    LunaLogCancellation => 5
    LunaLogWorkerExit => 6
    LunaLogWorkerProtocol => 7
    LunaLogNetworkRead => 8
    LunaLogNetworkWrite => 9
    LunaLogPeerClosed => 10
    LunaLogServiceDraining => 11
    LunaLogInternal => 12
  }
}

///|
fn luna_instance_log_reason_from_code(code : Int) -> LunaInstanceLogReason {
  match code {
    0 => LunaLogNoReason
    1 => LunaLogAuthentication
    2 => LunaLogValidation
    3 => LunaLogCapacity
    4 => LunaLogDeadline
    5 => LunaLogCancellation
    6 => LunaLogWorkerExit
    7 => LunaLogWorkerProtocol
    8 => LunaLogNetworkRead
    9 => LunaLogNetworkWrite
    10 => LunaLogPeerClosed
    11 => LunaLogServiceDraining
    _ => LunaLogInternal
  }
}

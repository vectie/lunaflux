///|
pub extend ExclusiveAdmissionCommit with @debug.Debug::{to_repr}

///|
pub extend ExclusiveAdmissionCommit with Eq::{not_equal, equal}

///|
pub extend SchedulerField with @debug.Debug::{to_repr}

///|
pub extend SchedulerField with Eq::{not_equal, equal}

///|
pub extend SchedulerIssue with @debug.Debug::{to_repr}

///|
pub extend SchedulerIssue with Eq::{not_equal, equal}

///|
pub extend SchedulerError with @debug.Debug::{to_repr}

///|
pub extend SchedulerError with Eq::{not_equal, equal}

///|
pub extend ExactDeadlineResult with @debug.Debug::{to_repr}

///|
pub extend ExactDeadlineResult with Eq::{not_equal, equal}

///|
pub extend ExactDeviceInvalidation with @debug.Debug::{to_repr}

///|
pub extend ExactDeviceInvalidation with Eq::{not_equal, equal}

///|
pub extend ExactInstanceLossDrain with @debug.Debug::{to_repr}

///|
pub extend ExactInstanceLossDrain with Eq::{not_equal, equal}

///|
pub extend RequestStatus with @debug.Debug::{to_repr}

///|
pub extend RequestStatus with Eq::{not_equal, equal}

///|
pub extend CompletionCommit with @debug.Debug::{to_repr}

///|
pub extend CompletionCommit with Eq::{not_equal, equal}

///|
pub extend InstanceLossDrain with @debug.Debug::{to_repr}

///|
pub extend InstanceLossDrain with Eq::{not_equal, equal}

///|
pub extend CancelReservationStart with @debug.Debug::{to_repr}

///|
pub extend CancelReservationStart with Eq::{not_equal, equal}

///|
pub extend CancelReservationCommit with @debug.Debug::{to_repr}

///|
pub extend CancelReservationCommit with Eq::{not_equal, equal}

///|
pub extend SchedulerSlotId with @debug.Debug::{to_repr}

///|
pub extend SchedulerSlotId with Eq::{not_equal, equal}

///|
pub extend SchedulerSlotId with Hash::{hash, hash_combine}

///|
pub extend SchedulerPublicationRoute with Eq::{not_equal, equal}

///|
pub extend SchedulerRequestState with @debug.Debug::{to_repr}

///|
pub extend SchedulerRequestState with Eq::{not_equal, equal}

///|
pub extend TerminalReason with @debug.Debug::{to_repr}

///|
pub extend TerminalReason with Eq::{not_equal, equal}

///|
pub extend SchedulerPublicationKind with @debug.Debug::{to_repr}

///|
pub extend SchedulerPublicationKind with Eq::{not_equal, equal}

///|
pub extend TerminalNotice with @debug.Debug::{to_repr}

///|
pub extend TerminalNotice with Eq::{not_equal, equal}

///|
pub extend GeneratedTokenNotice with Eq::{not_equal, equal}

///|
pub extend SchedulerPublication with Eq::{not_equal, equal}

///|
pub extend RequestHandle with @debug.Debug::{to_repr}

///|
pub extend RequestHandle with Eq::{not_equal, equal}

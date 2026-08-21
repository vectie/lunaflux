#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scripts/validate-luna-framed-event-allocations.sh

if rg -n \
    '^pub fn LunaFramedEvent(Workspace|Work|View)::(ack|retire|event|owner|epoch|storage|raw)\(' \
    service/framed_wire/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna framed outbound surface leaks semantic or raw transport authority' >&2
  exit 1
fi

if ! rg -q \
    '^pub fn LunaOnlineEventCredit::ack\(Self\) -> Unit raise LunaOnlineInstanceError$' \
    service/online_session/pkg.generated.mbti ||
  ! rg -q \
    '^pub fn LunaOnlineEventCredit::view\(Self\) -> @luna_event\.LunaEventView raise LunaOnlineInstanceError$' \
    service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'Luna online event credit surface drifted' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux framed outbound allocation and authority gate passed.'

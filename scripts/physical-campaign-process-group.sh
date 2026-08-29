#!/bin/sh

# Generic tracked process-group mechanics for evidence runners. Callers own
# timeout policy, result schemas, and success decisions. The process-group
# identity remains retained after the leader exits so a failed leader cannot
# orphan a worker descendant.

lunaflux_campaign_pid=
lunaflux_campaign_pgid=
lunaflux_last_campaign_pgid=

lunaflux_stop_campaign_group() {
  campaign_pid=${lunaflux_campaign_pid:-}
  campaign_pgid=${lunaflux_campaign_pgid:-}
  if [ -n "$campaign_pgid" ] && kill -0 -- "-$campaign_pgid" 2>/dev/null; then
    kill -TERM -- "-$campaign_pgid" 2>/dev/null || true
    for _campaign_cleanup_attempt in 1 2 3 4 5; do
      kill -0 -- "-$campaign_pgid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$campaign_pgid" 2>/dev/null; then
      kill -KILL -- "-$campaign_pgid" 2>/dev/null || true
      for _campaign_cleanup_attempt in 1 2 3 4 5; do
        kill -0 -- "-$campaign_pgid" 2>/dev/null || break
        sleep 1
      done
    fi
    if kill -0 -- "-$campaign_pgid" 2>/dev/null; then
      return 1
    fi
  fi
  if [ -n "$campaign_pid" ]; then
    wait "$campaign_pid" 2>/dev/null || true
  fi
  lunaflux_campaign_pid=
  lunaflux_campaign_pgid=
}

lunaflux_run_tracked_campaign() {
  campaign_stdout=$1
  campaign_stderr=$2
  shift 2
  command -v setsid >/dev/null 2>&1 || return 127
  setsid "$@" >"$campaign_stdout" 2>"$campaign_stderr" &
  lunaflux_campaign_pid=$!
  lunaflux_campaign_pgid=$lunaflux_campaign_pid
  lunaflux_last_campaign_pgid=$lunaflux_campaign_pgid
  campaign_status=0
  wait "$lunaflux_campaign_pid" || campaign_status=$?
  if kill -0 -- "-$lunaflux_campaign_pgid" 2>/dev/null; then
    if ! lunaflux_stop_campaign_group; then
      return 126
    fi
    [ "$campaign_status" -ne 0 ] || campaign_status=125
  else
    lunaflux_campaign_pid=
    lunaflux_campaign_pgid=
  fi
  return "$campaign_status"
}

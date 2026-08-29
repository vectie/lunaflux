#!/usr/bin/env bash

# Filesystem and identity admission shared only by the Qwen3 c32 preparation
# runner and its focused validator. Callers define fail() and usage().

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

is_sha256() { [[ $1 =~ ^[0-9a-f]{64}$ ]]; }

split_pinned() {
  local argument=$1
  [[ $argument == /*#sha256=* ]] || usage
  pinned_path=${argument%#sha256=*}
  pinned_sha=${argument##*#sha256=}
  is_sha256 "$pinned_sha" || fail 'a pinned digest is malformed'
}

require_pinned_file() {
  local argument=$1 label=$2
  split_pinned "$argument"
  [[ -f $pinned_path && ! -L $pinned_path &&
     $(realpath -- "$pinned_path") == "$pinned_path" ]] ||
    fail "$label is not a canonical regular file"
  [[ $(sha256_file "$pinned_path") == "$pinned_sha" ]] ||
    fail "$label digest differs"
}

require_executable() {
  require_pinned_file "$1" "$2"
  [[ -x $pinned_path ]] || fail "$2 is not executable"
}

require_new_output() {
  local target=$1 label=$2 parent name
  [[ $target == /* && ! -e $target && ! -L $target ]] ||
    fail "$label is not a new absolute path"
  parent=${target%/*}
  name=${target##*/}
  [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ &&
     -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" &&
     $target == "$parent/$name" ]] || fail "$label is not canonical"
}

field() {
  local name=$1 file=$2 value count
  value=$(sed -n "s/^$name=//p" "$file")
  count=$(grep -c "^$name=" "$file")
  [[ $count -eq 1 && -n $value ]] || fail "field is absent or duplicated: $name"
  printf '%s\n' "$value"
}

stop_server() {
  local pid=${server_pid:-} pgid=${server_pgid:-} leader_alive group_alive
  if [[ -n $pid ]]; then
    if kill -0 "$pid" 2>/dev/null; then
      # Normal shutdown signals only the lifecycle leader. Its trap owns the
      # ordered bridge close and native supervisor drain.
      kill -TERM "$pid" 2>/dev/null || true
    fi
    for _ in $(seq 1 300); do
      leader_alive=0
      group_alive=0
      kill -0 "$pid" 2>/dev/null && leader_alive=1
      if [[ -n $pgid ]] && kill -0 -- "-$pgid" 2>/dev/null; then
        group_alive=1
      fi
      [[ $leader_alive -eq 0 && $group_alive -eq 0 ]] && break
      sleep 0.1
    done
    # Failure cleanup may kill the exact owned process group if the leader
    # hung or exited without reaping its native/bridge children.
    if [[ -n $pgid ]] && kill -0 -- "-$pgid" 2>/dev/null; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    elif kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
  server_pid=
  server_pgid=
}

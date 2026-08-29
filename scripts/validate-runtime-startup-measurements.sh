#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

require() {
  pattern=$1
  shift
  if ! rg -q "$pattern" "$@"; then
    printf '%s\n' "missing runtime startup measurement boundary: $pattern" >&2
    exit 1
  fi
}

require 'LunaHttp1Metrics' service/http1/control_types.mbt service/http1/control_request.mbt
require 'b"/metrics"' service/http1/control_request.mbt
require 'lunaflux\.runtime\.startup\.v1' service/http1/control_response.mbt
require 'lunaflux\.runtime\.metrics\.v1' service/online_tcp/control_server_metrics.mbt
require 'duration > 86400000' service/http1/control_response.mbt
require 'publish_startup_measurement' service/online_tcp/control_server_prepare.mbt ops/runtime_instance/owner.mbt
require 'startup_measurement_published' service/online_tcp/control_server_types.mbt service/online_tcp/control_server_progress.mbt
require 'RuntimeStartupMeasurement' ops/runtime_instance/types.mbt ops/runtime_instance/owner_status.mbt
require 'cold_start_latency_millis:' cmd/lunaflux/native_run.mbt
require 'startup_measurement_readiness:' cmd/lunaflux/native_run.mbt

# Ordinary control turns still select immutable responses. The exact metrics
# route alone may render a live snapshot into startup-owned bounded backing.
if rg -q 'StringBuilder|@utf8\.encode|FixedArray::make|Bytes::make|\.to_string\(\)' \
  service/online_tcp/control_server_progress.mbt; then
  printf '%s\n' 'control reactor allocates or performs generic metrics formatting' >&2
  exit 1
fi
if [ "$(rg -c 'luna_online_render_control_metrics_response\(' \
    service/online_tcp/control_server_progress.mbt)" -ne 1 ] ||
  ! rg -q 'if route == LunaHttp1Metrics' \
    service/online_tcp/control_server_progress.mbt ||
  rg -q 'StringBuilder|@utf8\.encode|FixedArray::make|Bytes::make|\.to_string\(' \
    service/online_tcp/control_server_metrics.mbt ||
  ! rg -q 'LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY : Int = 4096' \
    service/online_tcp/control_server_metrics.mbt ||
  ! rg -q 'metrics_output : LunaOnlineTcpOutputScratch' \
    service/online_tcp/control_server_types.mbt; then
  printf '%s\n' 'live metrics escaped its exact bounded request-only renderer' >&2
  exit 1
fi

# The production metric document and CLI projection remain scalar-only.
if rg -q 'model|filesystem|credential|secret|path' \
  service/online_tcp/control_server_metrics.mbt; then
  printf '%s\n' 'live metrics response contains a sensitive dimension' >&2
  exit 1
fi

printf '%s\n' 'Runtime startup measurement boundaries are valid.'

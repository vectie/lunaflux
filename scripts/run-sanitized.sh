#!/bin/sh

set -eu

# High-entropy Linux ASLR can intermittently collide with the fixed shadow
# address range used by older distro sanitizer runtimes. Restrict no-ASLR to
# the probe process; production executables retain the host security policy.
if [ "$(uname -s)" = Linux ] && command -v setarch >/dev/null 2>&1; then
  exec setarch "$(uname -m)" -R "$@"
fi

exec "$@"

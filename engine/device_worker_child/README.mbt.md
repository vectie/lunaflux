# Startup-only device child

This native command support package consumes exactly one Configure frame and
one canonical bootstrap-source frame from the private inherited channel. It
calls the real `device_worker_bootstrap` composition, queries and exactly
matches the resulting readiness contract, and only then writes Ready.

After Ready it accepts no plan or completion traffic. It waits only for parent
EOF, deterministically closes the device-worker bootstrap owner, and returns
success. Decode, bootstrap, cleanup, readiness, write, unexpected-frame, EOF,
or owner-close failure returns payload-free failure evidence; the command exits
nonzero without writing diagnostics. Cleanup-required bootstrap authority is
retried once before process termination.

The EOF primitive is native and separately covered under AddressSanitizer for
clean close, one-byte traffic, partial frame-prefix traffic, timeout, and file-
descriptor balance.

The package reads no argv or environment configuration and owns no path. This
is not the steady-state worker loop. CPU tests prove phase ordering and the
invalid-configuration end-to-end gate proves no Ready publication; physical
CUDA success and request execution remain deferred promotion evidence.

# Serialized device worker child

This native command support package consumes exactly one Configure frame and
one canonical bootstrap-source frame from the private inherited channel. It
calls the real `device_worker_bootstrap` composition, queries and exactly
matches the resulting readiness contract, and only then writes Ready.

Before Ready it allocates one reusable plan owner, completion owner, input
buffer, and output buffer from authenticated startup limits. After Ready it
serially reads one bounded plan or clean EOF, validates the frame, acquires the
completion writer before device mutation, delegates complete execution and
retirement to `DeviceWorkerBootstrapOwner::execute_frame`, then copies and
writes the exact submitted completion. The root-bound parent admits only one
in-flight plan for this serialized child. The loop is designed around reusable
fixed storage. Its positive-controlled generated-C gate covers the repeated
serialized control/transport branch, while the genuine fake-device aggregate
gate covers the public `DeviceWorkerOwner::execute_frame` lifecycle and its
measured row/sampling shapes. Physical CUDA allocation and numerical evidence
remain promotion gates.

Clean EOF before a new prefix closes the owner and succeeds. Partial prefix or
payload EOF, malformed frames, writer/device/completion failures, and transport
failures are fail-stop: no later completion is published, the owner is closed,
and the command exits silently nonzero. Cleanup-required bootstrap authority is
retried once before process termination.

The frame-or-EOF prefix primitive waits without an idle deadline for the first
byte, then bounds the remaining prefix and payload. AddressSanitizer coverage
includes clean close, complete and partial prefixes, timeout, and descriptor
balance; public wrapper tests also prove partial-payload poisoning.

The package reads no argv or environment configuration and owns no path. CPU
tests prove startup and steady-state ordering plus the fail-stop fault matrix.
A positive-controlled release generated-C gate proves the serialized loop's
repeated success branch contains no MoonBit allocator call. The invalid-
configuration end-to-end gate proves no Ready publication; physical CUDA
success remains deferred promotion evidence.

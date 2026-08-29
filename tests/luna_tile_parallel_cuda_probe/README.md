# LunaTile parallel CUDA probe

This test-only package exports the exact generic serial oracle and parallel
SIMT candidate for `sm120`, preserving their canonical identities, source
digests, entry points, launch geometry, and authority-free status. Its physical
runner compiles both sources twice, compares deterministic artifacts, executes
the parallel candidate against the serial/scalar oracle, runs memcheck and
racecheck, checks resource closure, and seals the evidence directory.

The local exporter, fake-tool transaction tests, and boundary validators pass.
The runner has not executed on NVIDIA hardware. It therefore supplies no
physical correctness, sanitizer, performance, manifest, or promotion evidence;
`manifest_bindable=false` and promotion authority remains absent.

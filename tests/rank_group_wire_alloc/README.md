# Rank-group wire allocation probe

This release-native executable warms one complete rank-group plan, then runs
128 more exact two-rank cycles while allocation interception is active. The
measured window executes canonical frame encode, standalone load, direct
transcript acceptance, and combined load-and-accept over reused startup-owned
buffers. A C execution mask proves all four paths ran, and the transcript's
final sequence proves every cycle completed.

The harness first allocates a fixed array under the same interceptor as a
positive control. Any measured allocation, missing execution bit, missing
release-C symbol, or incomplete final sequence aborts the gate. It creates no
production FFI dependency; the repository boundary gate allowlists only this
test directory alongside the existing allocation probes.

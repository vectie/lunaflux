# Self-contained offline two-rank checkpoint runner

`scripts/run-deepseek-dspark-two-spark.mbtx` remains one standalone MoonBit
automation entry point despite exceeding 800 lines. It coordinates preparation,
two-host startup, resident ordinary/speculative queues, cancellation and result
collection using one shared run envelope. Splitting it into separate standalone
scripts now would duplicate the host/placement envelope and make executor and
collector modes easier to mismatch. The small explicit offline logit-capture
mode uses that same envelope rather than creating another full orchestrator.

This exception is confined to offline automation. Production decoder, stage
observation and post-retirement capture are separate focused MoonBit files;
no production package imports the orchestrator. It does not authorize new
token-path diagnostics or a global mutable runtime configuration.

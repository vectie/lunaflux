#!/bin/sh

# Compatibility launcher only: compile once through the authoritative MoonBit
# builder. The removed shell implementation rescanned inputs and compiled twice.
exec moon run "$(dirname -- "$0")/build-qwen3-reusable-fused-runtime.mbtx" "$@"

#!/bin/sh

# Compatibility launcher only. All parsing and materialization lives in the
# checksum-free MoonBit entry point; no shell pipeline or alternate packager.
exec moon run "$(dirname -- "$0")/materialize-qwen3-bf16-launch-core.mbtx" "$@"

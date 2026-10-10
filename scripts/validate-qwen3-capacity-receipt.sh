#!/bin/sh
# Capacity/layout checks replace the retired payload-authentication gate.
exec moon run "$(dirname -- "$0")/validate-qwen3-preparation.mbtx" capacity

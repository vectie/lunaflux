#!/bin/sh
# Compatibility launcher; developer tests live in MoonBit, outside startup.
exec moon run "$(dirname -- "$0")/validate-qwen3-preparation.mbtx" serving

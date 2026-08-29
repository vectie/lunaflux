#!/bin/sh
set -eu
case "$*" in
  *--query-gpu=uuid*) printf '%s\n' 'GPU-fake-sm120-qualification' ;;
  *--query-gpu=pci.bus_id*) printf '%s\n' '00000000:01:00.0' ;;
  *--query-gpu=name*) printf '%s\n' 'Fake SM120 GPU' ;;
  *--query-gpu=compute_cap*) printf '%s\n' '12.0' ;;
  *--query-gpu=driver_version*) printf '%s\n' '590.48.01' ;;
  *) exit 2 ;;
esac

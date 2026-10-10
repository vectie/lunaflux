# Checked integer table narrowing

This family-neutral startup package converts a structurally inspected little-endian
row-major I64 table into an explicitly releasable I32 sidecar. Every value is
range checked, every row is checked for duplicates, input/output extents are
bounded before allocation, and invalid partial output is discarded on failure.
Neither the source nor converted payload is hashed.
Release drops references and invalidates access without scrubbing the payload.

The package owns no model, filesystem, device, kernel, or execution authority.
It is a startup materialization primitive and must not be used in a token-step
path.

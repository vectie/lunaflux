# Checked integer table narrowing

This family-neutral startup package converts an authenticated little-endian
row-major I64 table into an explicitly releasable I32 sidecar. Every value is
range checked, every row is checked for duplicates, input/output extents are
bounded before allocation, and invalid partial output is zeroed before failure.

The package owns no model, filesystem, device, kernel, or execution authority.
It is a startup materialization primitive and must not be used in a token-step
path.

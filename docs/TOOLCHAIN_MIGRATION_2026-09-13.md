# MoonBit stable toolchain migration

Target: `moon 0.1.20260904 (94521db)` and
`moonc v0.10.12+1634b282e (2026-09-07)` on macOS and Linux.

The migration replaces removed generic `strconv.from_str` calls with typed
`string.parse_int`, `parse_int64`, or `parse_uint` calls. Existing error
translation and canonical spelling checks remain in place. Array construction
uses `Array(capacity=...)`, retaining each original capacity. Test-only imports
are scoped to their test builds; unused imports are removed.

No kernel schedule, numeric precision, model graph, or request scheduling policy
is changed. This compatibility migration alone is not a performance result.
Rebuild serving executables before comparing the new toolchain with the previous
benchmark; reusing old executables would not measure this migration.

Regression coverage includes signed integer extrema, overflow, and rejection of
noncanonical integer spelling in the evidence reader, alongside the existing
safetensors integer/range and tokenizer suites.

Local validation: warning-denied native check passed; full native suite passed
3,712/3,712, followed by the integer-reader package passing 5/5 including the
new regression. `moon info --target native` completed. The global formatter
check reports unrelated record-format differences under the new formatter;
this migration does not include a repository-wide formatting rewrite.
Native allocation-test interception headers also emit C attribute warnings
against the new runtime header; these did not fail the suite. Linux rebuild and
fresh serving throughput measurements remain separate follow-up validation.

# Warning-denied native validation — 20 November 2026

This follow-up closes the compiler migration work left open in the
[19 November publication record](PUBLICATION_VALIDATION_20261119.md).
Warning settings, numerical tolerances, fixture sizes and optional-asset
conditions were not weakened.

## Changes

- Declare formerly implicit derived trait methods explicitly, and regenerate
  the interfaces. Existing interfaces lose no declarations; the new bounded
  JSON helper is the only additional non-derived function.
- Qualify blackbox test references where required, remove redundant
  annotations where warning 73 requires it, and move test-only imports into
  their test scopes. Remove unused aliases and duplicate imports.
- Replace deprecated numeric conversions, collection ownership methods, JSON
  construction and byte literals with their current equivalents. Preserve
  arithmetic order, bit reinterpretation and expected numerical results.
- Replace local `try?` failure assertions with `catch`/`noraise` control flow.
  The same operations must still raise, and the surrounding lifecycle and
  resource-count assertions remain in place.
- Use scoped `errdefer` for resource release, execution invalidation and lease
  rollback. Keep callback failure precedence and original diagnostic errors.
  Remove redundant catch-and-rethrow wrappers from successful-path tests.
- Replace the deprecated JSON parser depth option with a constant-space
  lexical depth preflight followed by the core parser. The existing depth-32
  bounds remain enforced; quoted delimiters, escapes, duplicate-key behavior,
  malformed syntax and invalid limits have focused tests. Chat normalization
  retains its existing depth-16 duplicate-key guard.
- Split shared admission and transformer test fixtures into focused files.

## Results

`moon info`, `moon fmt`, and native checks with `--deny-warn` pass. The stronger
`--deny-warn --warn-list +73` check also passes on macOS and matched Linux
AArch64; the annotation validation script passes.

The final complete Linux native suite passes **5,546 tests, zero failures**:

~~~sh
moon test --target native --deny-warn --warn-list +73 -j 8
~~~

The ten existing optional model-asset notices are unchanged. Two new bounded
JSON parser tests account for the increase from the previous 5,544-test Linux
baseline; no test was removed or disabled.

The first full Linux run passed 5,545 of 5,546 tests. The remaining test used
the old display spelling for a read-only array. Its three expected layer
sequences are now checked by exact value equality, including their order,
rather than by display formatting. The 24-test decoder regression then passed
with warnings denied and warning 73 enabled.

The Linux run uses MoonBit `v0.10.15+943464bfa`, pinned dependencies, and Docker
image `sha256:5b1332bafb8e0bd31065cfd4a7ea2d57b81dd68f96c4cb6b4ebac477bf921b23`.
Networking is disabled and no hardware devices are passed through. The
validated source archive has SHA-256
`8c61a1253846716cf5513983bac793890fb201f0f93bfa41c94fdcdbc77c8be2`.
The tested code is committed through `9919d39436d81f82d95b54ee3d9a3f39bb2b55be`;
the archive precedes this validation documentation only.
The [evidence record](evidence/warning-denied-validation-20261120.json) binds
the results to the source and retained log hashes.

This qualifies the native software warning gates. Existing optional model
asset cases still report their original skip notices; this run makes no new
physical GPU, FPGA, NCCL or full-model numerical claim.

## Separate debt audit

Follow-up: [the structural debt closure](STRUCTURAL_DEBT_CLOSURE_20261120.md)
now passes the unchanged debt audit and its rejection tests. The paragraph
below records the earlier state before that refactor.

The supplementary whole-checkout debt audit remains non-clean. It scans local
generated/vendor artifacts and also finds pre-existing oversized tracked
files. For example, `cmd/deepseek_checkpoint/rank.mbt` was already 512 lines
and `cmd/lunachip_qwen/benchmark.mbt` was already 687 lines. The migrated
admission test that crossed the size limit was split back below it. These
structural findings do not represent compiler warnings and are not reported
as a passing debt-policy audit.

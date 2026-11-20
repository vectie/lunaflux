# Structural debt closure — 20 November 2026

This follows the [warning-denied qualification](WARNING_DENIED_VALIDATION_20261120.md)
and closes the separately recorded structural audit findings. The debt policy
and `scripts/validate-debt-policy.sh` remain byte-for-byte unchanged: MoonBit
files must stay below 500 logical lines, other first-party code at or below
800, and the marker, global-state and naming checks remain enforced.

## Maintained source

The initial audit reported 39 oversized tracked files, three unowned markers
inside the upstream stb header, and one vague filename.

- Split 34 MoonBit files into focused files in their existing packages. A
  preservation check confirms that all 510 declaration blocks remain exactly
  once and unchanged; public names, ownership, visibility and signatures stay
  the same.
- Extract vendor capture-budget reservation and two export-receipt printers
  from the two functions that exceeded a file budget by themselves. Inlining
  those private helpers reconstructs the original statement sequences.
- Separate device preparation, graph geometry/owners, transport and inherited
  frame IO, kernel recipes/lowering/admission, export arguments and receipts,
  request preparation, event offers/stalls and tokenizer vocabulary/merge work.
  Test fixture and test-case splits retain every test and assertion.
- Rename the block-FP8 `helpers.mbt` file to `numeric_decode_vocabulary.mbt`,
  preserving its bytes and CUDA numeric vocabulary.
- Place the complete stb implementation under `media/rgb8/third_party/stb`,
  with its license and existing JPEG fix unchanged. The original include path
  remains a three-line compatibility include. Maintained codec ownership code
  stays inside the first-party audit.
- Replace the 1,587-line advanced-model shell validator with a small MoonBit
  runner and explicit data retaining all 588 predicates and 107 validation
  packages. Production dependency checks distinguish test-only imports;
  hostile fixtures prove forbidden runtime imports still reject.
- Move canonical physical-campaign record verification into a bounded MoonBit
  helper. Twelve valid/hostile records agree with the original Bash oracle.
  The campaign's argument admission, stage ordering, process groups, sealing
  and exact evidence checks remain in place.
- Update source guards to inspect the declarations' new locations, including
  all newly separated hot-path files. Repair stale `Int` and environment-name
  anchors to match the existing `Int32` declarations and C implementation;
  no execution constant, native ABI or runtime implementation changes.

`moon info` reports no generated-interface changes. No CUDA arithmetic,
numerical tolerance, fixture size, lifecycle transition or test was weakened.

## Local generated artifacts

The initial whole-checkout audit also encountered 159 oversized files and 77
upstream/generated marker findings under ignored artifact trees. These are
preserved outside the maintained source tree rather than deleted, edited or
given new audit exemptions:

| Compatibility path | Preserved sibling artifact directory |
| --- | --- |
| `target` | `../lunaflux-artifacts/structural-debt-20261120/target` |
| `benchmarks/results` | `../lunaflux-artifacts/structural-debt-20261120/benchmark-results` |
| `benchmarks/qwen3_comparison/results` | `../lunaflux-artifacts/structural-debt-20261120/qwen3-comparison-results` |

Before relocation, Git confirmed that each tree contained no tracked files
and was already ignored. Same-filesystem renames preserve all artifact bytes;
local symbolic links keep existing producer and replay paths usable. The
external `relocations.json` records the exact destinations. The existing
ignore entries now handle directories and their compatibility links. No
absolute local link is committed, and `moon clean` cannot delete the archive.
A source-only Linux checkout also passes the unchanged debt audit.

## Validation

The unchanged debt audit passes on both the working checkout and a source-only
Linux copy. `moon info`, `moon fmt`, formatting checks and native checking pass
with warnings denied and warning 73 enabled. Generated public interfaces have
no diff.

The full Linux AArch64 native suite passes **5,546/5,546 tests**. The converted
advanced-model validator passes its full default flow: all **588 predicates**,
formatting and warning-denied checking of **107 packages**, and **824/824 native
tests**. The ten existing optional model-asset notices are unchanged; no new
skip condition was added.

The tested code is committed through
`96e13642b0495b887973f3602c584ead2a35d2d8`. The source archive has SHA-256
`0712fc6cf32f3d7f44a68d2b7851aee497e9d9609d540f714d6da2f0078794cf`;
subsequent closure documentation changes no tested code. The native run uses
the matched MoonBit `v0.10.15+943464bfa` toolchain. The converted-validator
container adds ripgrep to the same base image. Both runs disable networking
and pass no hardware devices through.

The [machine-readable record](evidence/structural-debt-validation-20261120.json)
retains the initial findings, declaration move inventory, successful raw logs,
source identity, environment identities and unchanged policy/auditor blob IDs.

The codec ASan/UBSan probe passes its PNG/JPEG, buffered-entropy, truncation and
allocation-balance checks. The debt-policy hostile self-tests, advanced-model
hostile fixtures, canonical-evidence equivalence, token-step/no-hashing,
physical-transport, process-ABI and current-source campaign boundary checks
also pass. These are software checks; no powered robot or new physical GPU
campaign was run.

Git's whitespace check reports two pre-existing trailing-whitespace lines in
the relocated third-party header. They remain unchanged to preserve upstream
bytes; no whitespace exception or debt-policy relaxation was added.

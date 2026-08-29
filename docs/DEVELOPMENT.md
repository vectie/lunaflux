# Local development check

Use one package-local command during implementation:

~~~sh
scripts/check-local.sh engine/device_step
~~~

Pass every affected package or MoonBit test file in the same invocation. For
example:

~~~sh
scripts/check-local.sh engine/device_step engine/device_worker
~~~

The wrapper only checks formatting, performs the warning-denied native type
check (including warning 73), and runs the selected MoonBit tests. It does not
invoke CUDA hardware probes, sanitizers, soaks, evidence aggregation, release
assembly, benchmark campaigns, or deployment/network approval gates.

Validation has three tiers:

1. Ordinary edits use the focused command above.
2. A changed native ABI, kernel, network, or release boundary adds only that
   boundary's relevant sanitizer, physical, static, or transaction gate.
3. Completed phase and release candidates run the aggregate commands and
   campaigns in `docs/PLAN.md`.

Documentation-only edits use Markdown/fence/stale-claim scans plus
`git diff --check`; they do not require MoonBit, sanitizer, hardware, soak, or
release campaigns. A source/static harness pass proves only composition and
fail-closed validation. It becomes physical evidence only when an immutable
campaign result names the exact source, hardware, duration where applicable,
resource balance, and terminal outcome.

Startup authentication and qualification remain outside the production token
step. Release-evidence packages do not control runtime dispatch, and unrelated
physical, soak, benchmark, or packaging campaigns must not block the normal
developer loop.

# Advanced model-family boundary inventory

`advanced_model_rules.jsonl` retains all 588 expanded structural predicates
from the former shell validator. `advanced_model_packages.json` retains its
107 formatting, warning-denied check and native-test package selections.
The MoonBit runner owns execution and error handling; the shell entry point
only forwards to it.

The rules retain required symbols, forbidden dependencies, exact manifest
counts, model-family separation, authority seams and the strict below-500-line
MoonBit budget. Dependency direction checks inspect production import blocks;
`for "test"` and `for "wbtest"` imports supply fixtures without granting
runtime authority. Missing files, unknown scopes and failed searches reject.

Run the full original validation flow from the repository root:

~~~sh
scripts/validate-advanced-model-family-boundary.sh
~~~

For a structural-only edit loop or the validator's hostile fixtures:

~~~sh
moon run scripts/validate-advanced-model-family-boundary.mbtx --static-only
moon run scripts/validate-advanced-model-family-boundary.mbtx --self-test
~~~

The self-test rejects removed required symbols, forbidden runtime imports,
search failures, 500-line files and unterminated 500-line files. Test-scoped
imports remain distinct from production imports. The default command retains
formatting, checking and native tests; `--static-only` is explicitly limited
to the structural predicates.

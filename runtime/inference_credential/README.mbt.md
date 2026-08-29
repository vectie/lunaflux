# Inherited inference credential

The opaque CLI accepts an optional deployment-created connected Unix stream at
fixed descriptor 6. A present descriptor carries one `LFC1KEY` frame, a
bounded nonempty credential, and write-side EOF. The owner closes the channel
and wipes its source buffer before publishing an opaque startup auth policy.
That policy has one idempotent close operation which overwrites its complete
fixed backing store and invalidates every verifier alias. OpenAI server and
connection-pool owners invoke it only after listener/service drain reaches
terminal closure; startup rejection paths invoke it before abandoning policy
ownership.

The authenticated instance policy remains authoritative for the accepted
credential-length ceiling. The credential is never sourced from argv,
environment variables, configuration files, or an implicit path, and it is
never combined with descriptor 5's drain protocol. This package owns no TLS,
public routing, tenant policy, or deployment authentication claim.

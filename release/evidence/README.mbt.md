# LunaFlux release evidence

This package owns the Phase 9 campaign declaration and host-verification
contract. It is inert: it opens no files, devices, sockets, containers, or
reviewer credentials, and it does not import a deployment adapter.

One strict `lunaflux.release-campaign.v1` record binds an exact release subject,
OCI image, physical target, every fixed evidence obligation, and the three
required reviewer roles. Evidence and reviews distinguish `Pending`, `Passed`,
`Failed`, and `NotRun`. `Passed` and `Failed` require their own strict-relative,
size-bounded, SHA-256-bound artifact. Pending or unrun work cannot carry a
terminal artifact. Reviewer terminal states additionally require an identity
digest and a separate attestation artifact. V1 also requires every evidence
and reviewer artifact to have unique content as well as a unique locator, so
independent obligations cannot collapse onto copies of one artifact.

The role set deliberately keeps runtime build, external-adapter build, direct
path, and external-adapter path evidence independent. The performance role
records total and losing workload counts; a campaign cannot omit losing
profiles merely because its overall performance review passes.

Canonical admission validates only the record bytes and independently supplied
record digest. Release verification is a separate step: the host supplies the
exact expected subject/image/target identities, allowlisted reviewer identity
digests, and independently observed artifact locators, sizes, and digests.
Successful verification returns an opaque capability that binds the campaign
digest and verified four-state result; the declared state alone is not that
capability. Missing
physical-target evidence, a missing reviewer attestation, an unobserved
terminal artifact, or an identity mismatch raises and can never become a
passing verdict. Signature policy, artifact acquisition, reviewer credentials,
and adapter implementation remain deployment-owned concerns.

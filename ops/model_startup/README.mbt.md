# Model startup preflight

This package owns the truthful boundary for operator model preflight. The
legacy `lunaflux run MODEL` form admits and closes the selected model root, then
stops at the first missing prerequisite: no independently supplied runtime-
descriptor digest exists in that form, so hardware is not probed out of order.

The explicit pinned form opens independent model and kernel roots, admits a
bounded descriptor and the complete existing inert model, weight, production
paged-execution, bootstrap, and `DeviceWorkerPlan` evidence, closes both roots,
then checks the exact assigned ordinal and target against CUDA inventory. A
matching device returns `LunaModelPreflightComplete`, which means only that the
diagnostic preflight finished. No preflight may translate that outcome into
Ready or start a listener; only a live device-worker owner can publish
readiness.

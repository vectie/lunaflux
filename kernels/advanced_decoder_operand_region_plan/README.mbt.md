# Advanced-decoder operand-region planning

This family-neutral package binds an admitted advanced-decoder operand ABI to
bounded allocation-relative region metadata. Every input names an operand
ordinal and role, an allocation identity digest, byte offset, and allocation
capacity. Planning requires exact ABI order and roles, aligned in-capacity
ranges, consistent capacities for repeated allocation identities, and no
overlap within an allocation.

The resulting plan is `InertUnqualified`. Allocation identities and offsets
are canonical metadata, not device pointers or allocation ownership. The plan
does not load a module, authenticate a live device, or grant launch authority.

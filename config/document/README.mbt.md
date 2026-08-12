# Configuration document reader

This package is the bounded startup adapter from a UTF-8 JSON document to the
focused `service`, `model`, and `device` records and their deterministic
`ResolvedPlan`. Runtime components should receive those focused records, not
the `ConfigDocument` envelope.

Both public entry points require an explicit positive byte limit. Byte input is
bounded before UTF-8 decoding; string input is validated and measured by its
encoded UTF-8 length before JSON parsing. Parser and focused-validation errors
are translated into bounded categories that never carry source JSON, field
names, paths, hostnames, or digests.

MoonBit core JSON represents objects as `Map[String, Json]` and replaces an
earlier value when a key is repeated. Before calling that parser, this reader
therefore runs `internal/json_guard`, a depth-bounded lexical structure pass
that decodes object keys and rejects duplicates, including escaped-equivalent
keys and duplicates inside nested objects or arrays. The canonical object is
then checked for every unknown or missing field.

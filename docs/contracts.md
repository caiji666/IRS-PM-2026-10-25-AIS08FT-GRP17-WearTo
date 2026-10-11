# Interface contracts v0.1

`TripRequest` stores user-provided facts. Leave missing fields unset and ask for clarification through the control layer. Do not invent budgets, dates, or weather conditions.

`FashionContext` separates hard requirements, soft preferences, and supporting evidence. An unknown attribute must not automatically satisfy a hard requirement.

`ProductCandidate` distinguishes owned items from new purchases. Represent money as integer minor units; owned items have zero purchase cost. Preserve unknown availability and attributes as unknown. All items involved in a price calculation must use the same currency. Currency conversion requires a separate implementation.

`OutfitResult` is the recommendation transfer structure. Its `constraint_status` defaults to `unverified`. Passing schema validation establishes structural validity only; it does not prove compliance with budget, availability, size, or weather requirements.

`ModuleResult` returns one of `ok`, `needs_input`, `no_candidates`, `not_implemented`, or `error`, together with `data` and `reason`. All current tool implementations are scaffolds returning `not_implemented`; this status must never be treated as success.

When updating a budget, change only fields explicitly supplied by the user. A request such as “make it cheaper” expresses a ranking preference or requires clarification; it must not produce an invented budget. Failure to find a solution within Top-K candidates means only that the current candidate pool has no solution.

Each module pull request should include serialized examples and document currencies, units, field meanings, provenance, and versions. Coordinate new required fields with upstream and downstream module owners before introducing them.

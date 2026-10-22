# Student 4 → Student 5 Local Integration and Test Simplification Design

## Goal

Use the current local working tree as the implementation baseline, connect Student 5 to the
latest dynamic Student 4 public service from `origin/main`, and simplify Student 5 tests to a
course-appropriate regression suite without discarding important business, security, privacy,
state, or module-boundary coverage. The resulting local changes must be ready for a later merge,
commit, and push to GitHub; this task stops before those Git operations only because the user has
explicitly requested no commit or push in this round.

## Constraints

- Do not commit or push.
- Preserve all existing local changes unless they are Student 5 tests deliberately simplified by
  this work.
- Do not merge or rebase the dirty branch during this implementation. Instead, validate the
  resulting patch against current `origin/main` in isolation so it is ready for later GitHub
  integration.
- Do not modify the divergent local `travel_agent/web/`, Student 1, or weather work.
- Do not describe cached output as a live five-module result.
- Do not require paid APIs for automated verification.

## Integration design

The public Student 4 boundary is `travel_agent.outfits.service.run(payload, *, model=None,
manifest=None)`. It accepts a dynamic Student 3 Phase 2 bundle and returns a `ModuleResult`.
The full-pipeline harness calls this public boundary and passes its successful `data` payload to
`travel_agent.recommendation.service.run` as the `student4` field. Student 5 continues to own
adaptation, hard constraints, final ranking, dialogue, confirmation, and preview; it does not
import the Student 4 model implementation directly.

The latest `origin/main` Student 4 implementation and its directly required files are ported into
the local tree without merging the branch. The ported files should remain identical to main where
possible so a later merge resolves cleanly.

## Currency behavior

Constraint comparison uses the request currency when every purchasable item already has that
same currency. No ECB snapshot is needed for this same-currency path. Currency conversion is used
only when one or more item currencies differ from the request currency. Conversion remains
explicit in the result metadata and must fail safely when a required rate is unavailable.

This rule lets a USD request with all-USD Student 4 products run offline and avoids a meaningless
USD → SGD → USD round trip.

## Test strategy

The suite is not optimized for 100% branch coverage or an arbitrary test count. Keep one or a
small parameterized set for each independently important behavior:

- budget, availability, reliable size, currency, no-candidate conflict, and `min_budget` rules;
- ranking formula, beta endpoints, deterministic ties, and Top-3;
- the latest Student 4 output schema, provenance, price, stock, score, and required categories;
- modification, rerun planning, stale recommendations, exact-ID confirmation, and preview gating;
- portrait consent, provider fallback, output validation, and representative SSRF protections;
- Student 1→2→3→4→5 payload boundaries and first-failure propagation;
- cached real-data Student 4→5 execution and separately labelled live smoke tests.

Delete or consolidate exhaustive framework-type permutations, internal parser branches, repeated
round trips, repeated immutability assertions, environment-variable combinations, equivalent
provider exception classes, and the same validation repeated at every layer. The expected final
size is approximately 280–340 collected tests, but correctness takes priority over the number.

## Safety and local-state handling

Before editing, record the current untracked and modified paths. Changes are limited to Student 4
public-interface files ported from main, Student 5 integration/currency code, the full-pipeline
harness, and directly related tests. Divergent web and Student 1 files remain untouched.

Verification reports Student 5 results separately from unrelated full-suite baseline failures.
The known snapshot newline failure and unrelated untracked Student 1/web failures are not hidden or
reported as passing.

## Acceptance criteria

1. A dynamic Student 3 bundle can pass through the latest Student 4 public service into Student 5.
2. Same-currency requests do not require FX data; mixed currencies still use validated FX data.
3. Core Student 5, contract, full-pipeline, main-agent, and cached-real-data tests pass offline,
   apart from explicitly documented optional skips.
4. The final Student 5 suite is materially smaller than the 468-test branch-coverage restoration
   while preserving the categories above.
5. Ruff, format checking, and `git diff --check` pass for in-scope files.
6. No commit, push, merge, rebase, or overwrite of divergent web/Student 1 work occurs.

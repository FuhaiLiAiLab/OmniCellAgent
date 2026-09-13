# Omics workflow review: deferred and unresolved work

This file records the review findings that are **not** resolved by the current
scoped change. The two items under “Current scoped work” are implemented here;
everything in the deferred register remains open until its acceptance criteria
are met and verified.

Priority legend:

- **P0** — can invalidate scientific conclusions, report a failed run as
  successful, or destroy/contaminate results.
- **P1** — material reliability, reproducibility, or integration risk.
- **P2** — compatibility and coverage work that should follow the correctness
  fixes.

## Current scoped work — implemented in this change

| Item | Status in this scope | Acceptance criterion |
| --- | --- | --- |
| Obsolete NER removal | **Implemented here; not deferred.** Remove the obsolete NER path from the primary workflow only. | The primary workflow has no import, registration, prompt, or dispatch path for the obsolete NER tool, and its focused tests pass. Alternate wrappers and MCP surfaces are tracked separately below. |
| CellAgent shadow root for CellTOSG `last_query_result.csv` | **Implemented here; not deferred.** Add a temporary writable shadow of the immutable CellTOSG input root, without changing CellTOSG. | Required CellTOSG inputs are read through explicit symlinks; the incidental `last_query_result.csv` write remains inside temporary storage; real outputs continue to use the session directory; the shared dataset root is unchanged; and temporary files are removed after loading. |

## Deferred issue register

### P0 — scientific validity and truthful outcomes

1. **Cell-as-replicate analysis causes pseudoreplication; donor-aware
   pseudobulk is absent.** Cells from the same donor/sample must not be treated
   as independent biological replicates for inferential differential
   expression.
   - Acceptance: the default inferential path aggregates counts at the
     donor/sample-by-condition level, fits a donor-aware pseudobulk design,
     validates the minimum number of independent replicates, and refuses an
     under-replicated comparison with an explicit diagnostic. If a cell-level
     exploratory analysis remains available, it is clearly labelled
     non-inferential and cannot be mistaken for the default result.

2. **Genuine cell QC is missing or insufficient.** Merely accepting a matrix or
   filtering labels is not cell-level quality control.
   - Acceptance: the workflow applies and records explicit, configurable QC
     metrics and thresholds (for example detected genes, library size, and
     mitochondrial fraction where available), reports cells retained and
     removed by reason and sample, and performs QC before aggregation and
     testing. Missing metrics must produce an explicit warning or failure, not
     an implicit pass.

3. **Unknown/unclassified control handling is ambiguous.** Unknown,
   unclassified, missing, or mixed control labels can enter the reference group
   or be dropped silently.
   - Acceptance: control classification is explicit and deterministic;
     unknown/unclassified values are rejected or handled only through a named
     user policy; counts and exclusions are reported; and no unknown label is
     silently coerced into a control.

4. **Automatic label fallback can change the estimand.** Falling back from a
   requested condition/cell-type/donor label to another field or broader
   population silently changes the biological comparison.
   - Acceptance: a missing or unusable requested label fails with the available
     choices and a clear remediation message. Any fallback must be explicitly
     selected by the caller and recorded in outputs; the resulting contrast
     and analysis population are printed before execution.

5. **The configured log2 fold-change threshold is ignored.** A displayed or
   accepted `log2FC` cutoff that is not used changes gene selection without the
   user knowing.
   - Acceptance: the validated threshold is applied consistently to both
     up- and down-regulated gene selection, propagated to enrichment inputs,
     included in the report/provenance, and covered by boundary-value tests.

6. **Enrichment background universe and significance semantics are
   underspecified.** Using an implicit whole-database background, raw p-values,
   or inconsistent thresholds can make enrichment claims misleading.
   - Acceptance: the tested-gene universe is explicit and passed to every
     supported backend when possible; unsupported backends disclose their
     background; multiple-testing correction and the significance cutoff are
     defined and applied consistently; empty/non-significant results are
     distinguished from failures; and reports include gene-set/database
     versions.

7. **R/KEGG execution can report false success, and artifacts are not verified.**
   An R process or KEGG step may fail, emit an error-shaped response, or omit
   outputs while the wrapper continues as though it succeeded.
   - Acceptance: process exit status, captured stderr, R-side error state, and
     KEGG response semantics are checked; expected tables/plots are verified to
   exist, be newly produced for the current run, parse successfully, and meet
   minimal schema/content checks; “no enriched terms” is a valid explicit
   outcome; and an R/KEGG-only failure preserves valid upstream DE/enrichment
   as a visibly partial result with a warning rather than failing the whole run.

8. **CLI exit codes and reported status are not guaranteed truthful.** A failed
   stage must not end with exit code zero or a success message/result envelope.
   - Acceptance: every terminal failure returns a non-zero exit code and a
     machine-readable failed status, success is emitted only after required
     artifact verification, and cancellation/partial completion are distinct
     states tested end to end.

9. **Session paths, per-run isolation, and stale artifacts are unsafe.** Session
   identifiers/paths need validation, and mutable shared filenames can overwrite
   or reuse output from another run.
   - Acceptance: all resolved input/output paths are validated to remain inside
     an allowed session root (including traversal and symlink checks); every run
     writes to an immutable unique run directory; required artifacts are tied
     to that run; pre-existing or stale files cannot satisfy success checks;
     and cross-session reads/writes fail closed.

10. **Background destructive cleanup can race with work or delete evidence.**
    Detached or broadly targeted cleanup can remove another run’s artifacts,
    erase failure evidence, or race with readers.
    - Acceptance: no untracked background deletion is used; cleanup targets a
      validated, explicit run directory; active runs are protected; retention
      is configurable; cleanup failures are surfaced; and tests prove that
      sibling/session-root paths cannot be deleted.

### P1 — robust failure handling and reproducibility

11. **Errors are not structured or typed across workflow boundaries.** Free-form
    strings and generic exceptions prevent callers from distinguishing invalid
    input, no result, retryable infrastructure failure, and terminal analysis
    failure.
    - Acceptance: Python, R/subprocess, CLI, and tool boundaries use a stable
      error envelope with a typed code/category, stage, human-readable message,
      retryability, relevant safe context, and chained cause; callers branch on
      the type/code rather than parsing text.

12. **Enrichment API transport and response failures are incomplete.** The
    workflow needs separate handling for connection timeout, read timeout,
    DNS/other network errors, non-success HTTP responses (including rate limits
    and server failures), malformed JSON/body/schema, and partial results when
    only some requested databases succeed.
    - Acceptance: connect and read timeouts are finite and separately
      configurable; HTTP status and retry guidance are preserved; network and
      malformed-response errors map to typed failures; retries are bounded and
      limited to safe retryable cases; per-database status is retained; a
      partial-database result is visibly marked partial and names every failed
      database rather than appearing complete or empty.

13. **Agent tool failures are not reliably propagated into retry/replan logic.**
    Tool errors can be flattened, ignored, or turned into a plausible final
    answer without a bounded recovery policy. This remediation requires changes
    to `langgraph_agent.py` and is **explicitly paused/deferred outside the
    current scope**.
    - Acceptance when resumed: structured tool failures enter agent state;
      retryable failures use a bounded retry policy; non-retryable or exhausted
      failures cause an explicit replan or terminal failed status; the final
      answer cannot claim success after a failed required tool; and tests cover
      retry, replan, exhaustion, and mixed/partial tool outcomes.

14. **A provenance manifest is absent or incomplete.** Results cannot be fully
    reproduced without recording the exact inputs and execution context.
    - Acceptance: each immutable run emits a machine-readable manifest with run
      and session IDs, timestamps, input paths and content fingerprints,
      software/package and database versions, command/configuration, seeds,
      selected label columns and contrasts, QC/DE/enrichment thresholds,
      background universe, per-stage status, and artifact fingerprints.

### P2 — compatibility and verification coverage

15. **Alternate wrappers and MCP interfaces may still expose stale NER
    contracts.** Removing obsolete NER from the primary path does not update
    secondary CLI wrappers, adapters, manifests/schemas, prompts, examples, or
    MCP tool surfaces.
    - Acceptance: inventory all public and alternate entry points; remove or
      deliberately version/deprecate stale NER symbols and schemas; add a clear
      compatibility error where old clients cannot be supported; and run
      contract tests proving advertised tools match callable implementations.

16. **Test coverage does not exercise the review’s scientific and failure
    modes.** Happy-path unit tests alone cannot establish correctness here.
    - Acceptance: add focused unit, integration, and end-to-end tests for
      donor-aware pseudobulk and under-replication; QC and label edge cases;
      estimand-preserving label selection; `log2FC` boundaries; enrichment
      universe/significance; connect/read timeout, HTTP, network, malformed and
      partial-database responses; R/KEGG false-success and artifact validation;
      truthful CLI status/exit codes; typed error propagation; session
      traversal, symlink, stale-output and concurrent-run isolation;
      cleanup-scope safety; provenance completeness; paused agent recovery once
      `langgraph_agent.py` work resumes; and stale wrapper/MCP NER contracts.

## Completion rule

An item remains open until its implementation, automated tests, and user-facing
behavior satisfy the listed acceptance criteria. A warning or documentation-only
change does not close a correctness, safety, or truthfulness issue.

---
name: compare-telemetry
description: >
  Use when comparing OpenTelemetry traces, metrics, and logs between two versions
  of a GenAI instrumentation library (original vs migrated). Runs example scripts
  in both repos, captures telemetry via local OTel Collector file exporters, and
  generates a human-readable diff report with trace links, span attribute comparison,
  log event analysis, and metrics diff.
---

# Compare Telemetry

Compare OpenTelemetry telemetry (spans, metrics, logs) between two instrumentation
versions to validate a migration produces equivalent or intentionally-changed output.

## When to Activate

Use this skill when:
- Migrating a GenAI instrumentation library to a new pattern (e.g., TelemetryHandler/LLMInvocation)
- Validating that a PR preserves existing telemetry structure
- Comparing traces, metrics, and logs between two instrumentation versions
- Generating a diff report for human review of instrumentation changes

## Prerequisites

| Tool | Purpose | Verify |
|------|---------|--------|
| `python3` | Run comparison script and example apps | `python3 --version` |
| `pip` | Install dependencies | `pip --version` |
| `dotenv` | Load .env for example runs | `pip show python-dotenv` |
| OTel Collector | Local collector with file exporters | `lsof -i :4317` |

## Inputs

Only 4 inputs are required. Everything else is auto-deduced.

### Required (ask user)

| Input | Description | Example |
|-------|-------------|---------|
| `repo_original` | Path to original instrumentation repo | `$HOME/repo/splunk-otel-python-contrib3` |
| `repo_migrated` | Path to migrated instrumentation repo | `$HOME/repo/splunk-otel-python-contrib4` |
| `example_script` | Relative path to the test script | `instrumentation-genai/opentelemetry-instrumentation-google-genai/examples/manual/main2.py` |
| `pr_stage` | PR number being validated | `PR1` |

### Auto-deduced (do NOT ask user)

| Derived Input | Deduction Logic |
|---------------|-----------------|
| `service_name` | Read `.env.example` next to `example_script` and extract the `OTEL_SERVICE_NAME` value |
| `service_original` | `{service_name}-pr0` |
| `service_migrated` | `{service_name}-pr{N}` (N from `pr_stage`) |
| `scenarios` | Read the plan file (`.local/plan-*-migration.md` or `.local/plan-telemetry-comparison-skill.md`) and look up the **PR to Scenario Mapping** table for the given `pr_stage` |
| `collector_dir` | Use remembered default. If not set, ask once and remember for future runs |
| `realm` | Use remembered default (`shw-playground`). If not set, ask once and remember |
| `output_file` | `.local/comparisons/{framework}-{pr_stage_lower}.md` where framework is extracted from the package directory name (e.g., `google-genai` from `opentelemetry-instrumentation-google-genai`) |

### Deduction examples

Given `example_script = instrumentation-genai/opentelemetry-instrumentation-google-genai/examples/manual/main2.py` and `pr_stage = PR1`:

```text
framework       = google-genai   (from directory name, strip "opentelemetry-instrumentation-")
service_name    = opentelemetry-python-google-genai   (from .env.example)
service_original = opentelemetry-python-google-genai-pr0
service_migrated = opentelemetry-python-google-genai-pr1
scenarios       = simple,system_config,multi_turn,async_basic   (from plan file PR1 row)
output_file     = .local/comparisons/google-genai-pr1.md
```

## Limitations

- Only supports OTLP JSON file exporter format (line-delimited JSON)
- Requires both repos to share the same example script structure
- macOS-specific collector binary path (`otelcontribcol_darwin_arm64`)
- Comparison is scenario-order dependent (spans matched by position, not content)
- Non-deterministic fields are skipped entirely, not normalized

## Workflow

```text
Pre-flight --> Run original --> Run migrated --> Compare --> Report
   |               |                |               |          |
   v               v                v               v          v
 Check collector  Record offset   Record offset   Parse JSON  Trace links
 Set env vars    Execute script  Execute script  Diff attrs   Attribute diff
 Comment scenarios Wait flush     Wait flush      Diff events  Event diff
                                                  Diff metrics Metrics diff
```

### Phase 1: Pre-flight

1. Verify collector is running: `lsof -i :4317`
   - If NOT running, start **with log capture** (tee shows on terminal AND writes to file):
     ```bash
     cd <collector_dir>
     source .env && ./bin/otelcontribcol_darwin_arm64 --config ./collector-config.yaml 2>&1 | tee collector.log
     ```
   - If already running without `tee`, ask user to restart with the command above

2. Confirm scenarios in the example script — comment out any out-of-scope scenarios in both repos

3. Auto-deduce and set `OTEL_SERVICE_NAME` in both repos' `.env` files:
   - Read `.env.example` next to example_script to get base service name
   - Original: `{service_name}-pr0`
   - Migrated: `{service_name}-pr{N}`

4. Ensure dependencies installed in both venvs:
   ```bash
   pip install -e "./instrumentation-genai/<package>" "python-dotenv[cli]"
   ```

### Phase 2: Archive and reset export files

5. Archive any existing export data with a timestamp so it can be referred back to if needed:
   ```bash
   TS=$(date +%Y%m%d_%H%M%S)
   ARCHIVE_DIR=<collector_dir>/archive/$TS
   mkdir -p $ARCHIVE_DIR
   cp <collector_dir>/traces-export.json $ARCHIVE_DIR/traces-export.json 2>/dev/null || true
   cp <collector_dir>/metrics-export.json $ARCHIVE_DIR/metrics-export.json 2>/dev/null || true
   echo "Archived to $ARCHIVE_DIR"
   ```
   Then truncate both files so only the current run's data is present:
   ```bash
   truncate -s 0 <collector_dir>/traces-export.json
   truncate -s 0 <collector_dir>/metrics-export.json
   ```

### Phase 3: Run original (baseline)

6. Run: `source <repo_original>/.venv/bin/activate && dotenv -f <example_dir>/.env run -- python <script>`
7. Wait 20 seconds for telemetry to flush
8. Verify file grew: `wc -c < <collector_dir>/traces-export.json`

### Phase 4: Run migrated

9. Run: `source <repo_migrated>/.venv/bin/activate && dotenv -f <example_dir>/.env run -- python <script>`
10. Wait 20 seconds for telemetry to flush
11. Verify file grew: `wc -c < <collector_dir>/traces-export.json`

### Phase 5: Generate comparison report

12. Run the comparison script:
    ```bash
    python3 skills/compare-telemetry/scripts/compare_telemetry.py \
      --traces-file <collector_dir>/traces-export.json \
      --metrics-file <collector_dir>/metrics-export.json \
      --service-original <service_original> \
      --service-migrated <service_migrated> \
      --scenarios <scenarios> \
      --realm <realm> \
      --output <output_file>
    ```

### Phase 6: Verify and review

13. Check collector logs for export errors:
    ```bash
    grep -i 'error\|fail\|401\|403' <collector_dir>/collector.log | grep -v otlphttp/logs | tail -10
    ```

14. Present the trace links table to the user for UI review

15. Present the comparison summary:
    - [PRESERVED] — same in both
    - [CHANGED] — expected per migration plan
    - [REGRESSION] — unexpected difference, needs fix

## Non-Deterministic Fields

The comparison script ignores these attributes (vary between runs):

- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.input.messages`, `gen_ai.output.messages`
- `gen_ai.system_instructions`
- `gen_ai.response.id`
- `gen_ai.evaluation.sampled`, `gen_ai.evaluation.error`
- All trace/span IDs, timestamps, durations

## Known Issues

| Issue | Solution |
|-------|----------|
| Collector port in use | Kill stale process: `lsof -i :8888 -t \| xargs kill; lsof -i :4317 -t \| xargs kill` |
| Traces not in O11y | Check `collector.log` for SAPM errors. Verify realm matches browser org |
| Wrong realm in links | Use the realm from a known-working trace URL, not from collector config |
| Logs export 404 | Expected — `otlphttp/logs` to `ingest.us1.signalfx.com/v1/logs` is unsupported |
| `Part.from_text()` error | Use keyword arg: `Part.from_text(text="...")` |
| Flat trace links useless | All gen_ai spans have same name — always use Scenario × PR table |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No spans extracted | Check `service.name` matches exactly (case-sensitive) |
| Missing metrics | Ensure `metrics-export.json` exists and collector has `file` exporter in metrics pipeline |
| Wrong scenario count | Verify example script has correct scenarios enabled/commented |
| Script exits with 0 spans | File may contain stale data — check offset grew after run |

## Examples

Run the comparison script directly:

```bash
python3 scripts/compare_telemetry.py \
  --traces-file $COLLECTOR_DIR/traces-export.json \
  --metrics-file $COLLECTOR_DIR/metrics-export.json \
  --service-original my-service-pr0 \
  --service-migrated my-service-pr1 \
  --scenarios simple,system_config,async_basic,multi_turn \
  --realm shw-playground \
  --output .local/comparisons/my-comparison.md
```

Expected output:

```text
Parsed: 4 original spans, 4 migrated spans
Events: 12 original, 8 migrated
Metrics: 6 original, 6 migrated
Report written to .local/comparisons/my-comparison.md
```

See `references/example-report-structure.md` for the expected report structure.

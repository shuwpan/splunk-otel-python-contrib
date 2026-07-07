#!/usr/bin/env python3
"""
Telemetry comparison tool for OpenTelemetry instrumentation migration validation.

Reads OTLP JSON file-exporter output (traces-export.json, metrics-export.json),
extracts spans/events/metrics for two service names, and generates a markdown
comparison report with pretty-print trace trees, attribute diffs, and trace links.

Usage:
    python compare_telemetry.py \
        --traces-file /path/to/traces-export.json \
        --metrics-file /path/to/metrics-export.json \
        --service-original opentelemetry-python-google-genai-pr0 \
        --service-migrated opentelemetry-python-google-genai-pr1 \
        --scenarios simple,system_config,async_basic,multi_turn \
        --realm shw-playground \
        --output /path/to/comparison.md
"""

import argparse
import json
import sys
from pathlib import Path

# Attributes to skip in comparison (non-deterministic)
SKIP_KEYS = {
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.response.id",
    "gen_ai.evaluation.sampled",
    "gen_ai.evaluation.error",
}


def parse_attr_value(value: dict):
    """Extract a Python value from an OTLP attribute value dict."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [
            list(x.values())[0] for x in value["arrayValue"].get("values", [])
        ]
    return str(value)


def parse_spans(filepath: str, service_filter: str) -> list[dict]:
    """Extract gen_ai spans for a given service from traces-export.json."""
    spans = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or service_filter not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rs in obj.get("resourceSpans", []):
                svc = ""
                for a in rs.get("resource", {}).get("attributes", []):
                    if a["key"] == "service.name":
                        svc = a["value"].get("stringValue", "")
                if svc != service_filter:
                    continue
                for ss in rs.get("scopeSpans", []):
                    scope = ss.get("scope", {}).get("name", "")
                    for s in ss.get("spans", []):
                        name = s.get("name", "")
                        if name.startswith(
                            ("POST", "GET", "PUT", "DELETE", "PATCH")
                        ):
                            continue
                        attrs = {
                            a["key"]: parse_attr_value(a["value"])
                            for a in s.get("attributes", [])
                            if a.get("value")
                        }
                        spans.append(
                            {
                                "name": name,
                                "attrs": attrs,
                                "scope": scope,
                                "traceId": s.get("traceId", ""),
                                "events": [
                                    e.get("name", "")
                                    for e in s.get("events", [])
                                ],
                            }
                        )
    return spans


def parse_metrics_and_logs(filepath: str, service_filter: str) -> dict:
    """Extract metrics and log events for a given service from metrics-export.json."""
    result = {"metrics": [], "events": []}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or service_filter not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "resourceMetrics" in obj:
                for rm in obj["resourceMetrics"]:
                    svc = ""
                    for a in rm.get("resource", {}).get("attributes", []):
                        if a["key"] == "service.name":
                            svc = a["value"].get("stringValue", "")
                    if svc != service_filter:
                        continue
                    for sm in rm.get("scopeMetrics", []):
                        scope = sm.get("scope", {}).get("name", "")
                        for m in sm.get("metrics", []):
                            dp_attrs = {}
                            for dtype in ("histogram", "sum", "gauge"):
                                for dp in m.get(dtype, {}).get(
                                    "dataPoints", []
                                ):
                                    dp_attrs = {
                                        a["key"]: list(a["value"].values())[0]
                                        for a in dp.get("attributes", [])
                                    }
                                    break
                            result["metrics"].append(
                                {
                                    "scope": scope,
                                    "name": m["name"],
                                    "dp_attrs": dp_attrs,
                                }
                            )

            elif "resourceLogs" in obj:
                for rl in obj["resourceLogs"]:
                    svc = ""
                    for a in rl.get("resource", {}).get("attributes", []):
                        if a["key"] == "service.name":
                            svc = a["value"].get("stringValue", "")
                    if svc != service_filter:
                        continue
                    for sl in rl.get("scopeLogs", []):
                        scope = sl.get("scope", {}).get("name", "")
                        for lr in sl.get("logRecords", []):
                            event_name = lr.get("eventName", "")
                            if not event_name:
                                for a in lr.get("attributes", []):
                                    if a["key"] == "event.name":
                                        event_name = a["value"].get(
                                            "stringValue", ""
                                        )
                            result["events"].append(
                                {
                                    "scope": scope,
                                    "event_name": event_name,
                                    "traceId": lr.get("traceId", ""),
                                }
                            )
    return result


# Semconv attributes that distinguish independent metric series within the same metric name.
# Each unique combination of these values is a separate series and must not be merged.
# Ref: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
_SERIES_DISCRIMINATORS = {
    "gen_ai.client.operation.duration": "gen_ai.operation.name",
    "gen_ai.client.token.usage": "gen_ai.token.type",
    "gen_ai.client.operation.time_to_first_chunk": "gen_ai.operation.name",
}


def metric_summary(metrics: list[dict]) -> dict:
    """Deduplicate metrics by (name, discriminating_dim), collecting unique dimension keys."""
    by_key = {}
    for m in metrics:
        name = m["name"]
        if name.startswith(("otel.sdk", "http.")):
            continue
        discriminator = _SERIES_DISCRIMINATORS.get(name)
        disc_value = (
            m["dp_attrs"].get(discriminator, "") if discriminator else ""
        )
        key = (name, disc_value)
        if key not in by_key:
            by_key[key] = {
                "scope": m["scope"],
                "name": name,
                "disc_value": disc_value,
                "dp_attr_keys": set(),
            }
        by_key[key]["dp_attr_keys"].update(m["dp_attrs"].keys())
    return by_key


def generate_report(
    spans_orig: list[dict],
    spans_migr: list[dict],
    ml_orig: dict,
    ml_migr: dict,
    scenarios: list[str],
    svc_orig: str,
    svc_migr: str,
    realm: str,
) -> str:
    """Generate the full markdown comparison report."""
    base_url = f"https://{realm}.signalfx.com/#/apm/traces"
    R = []

    # Header
    R.append(f"# Telemetry Comparison: {svc_orig} vs {svc_migr}\n")
    R.append(f"**Original:** `{svc_orig}`")
    R.append(f"**Migrated:** `{svc_migr}`")
    R.append(f"**Scenarios:** {', '.join(scenarios)}\n")

    # Trace links table
    R.append("## Trace Links (Splunk O11y)\n")
    R.append("| Scenario | Original | Migrated |")
    R.append("|----------|----------|----------|")
    for i, sc in enumerate(scenarios):
        orig_url = (
            f"{base_url}/{spans_orig[i]['traceId']}"
            if i < len(spans_orig)
            else "_(missing)_"
        )
        migr_url = (
            f"{base_url}/{spans_migr[i]['traceId']}"
            if i < len(spans_migr)
            else "_(missing)_"
        )
        orig_link = (
            f"[Original {sc}]({orig_url})"
            if i < len(spans_orig)
            else "_(missing)_"
        )
        migr_link = (
            f"[Migrated {sc}]({migr_url})"
            if i < len(spans_migr)
            else "_(missing)_"
        )
        R.append(f"| {sc} | {orig_link} | {migr_link} |")
    R.append("")

    # Pretty-print trace trees
    R.append("---\n")
    R.append("## Pretty-Print Trace Trees\n")
    for label, spans, events in [
        ("Original", spans_orig, ml_orig["events"]),
        ("Migrated", spans_migr, ml_migr["events"]),
    ]:
        R.append(f"### {label}\n")
        R.append("```")
        for i, s in enumerate(spans):
            sc = scenarios[i] if i < len(scenarios) else f"span_{i}"
            R.append(f"=== Scenario: {sc} ===")
            R.append(f"Trace: {s['traceId']}")
            R.append(f"Scope: {s['scope']}")
            R.append(f"└─ {s['name']}")
            for k, v in sorted(s["attrs"].items()):
                if k in SKIP_KEYS:
                    continue
                R.append(f"   ├─ {k} = {v}")
            matched = [e for e in events if e["traceId"] == s["traceId"]]
            for e in matched:
                R.append(f"   📨 {e['event_name']}")
            R.append("")
        R.append("```\n")

    # Span attribute comparison
    R.append("---\n")
    R.append("## Span Attribute Comparison\n")
    for i, sc in enumerate(scenarios):
        R.append(f"### Scenario: {sc}\n")
        if i >= len(spans_orig) or i >= len(spans_migr):
            R.append("⚠️ Span missing in one run\n")
            continue
        s0, s1 = spans_orig[i], spans_migr[i]
        R.append("| Attribute | Original | Migrated | Status |")
        R.append("|-----------|----------|----------|--------|")
        status = "✅ Same" if s0["name"] == s1["name"] else "❌ Different"
        R.append(f"| span name | `{s0['name']}` | `{s1['name']}` | {status} |")
        status = "✅ Same" if s0["scope"] == s1["scope"] else "🔄 Changed"
        R.append(f"| scope | `{s0['scope']}` | `{s1['scope']}` | {status} |")
        all_keys = sorted(
            set(list(s0["attrs"].keys()) + list(s1["attrs"].keys()))
            - SKIP_KEYS
        )
        for k in all_keys:
            v0, v1 = s0["attrs"].get(k), s1["attrs"].get(k)
            if v0 == v1:
                R.append(f"| `{k}` | `{v0}` | `{v1}` | ✅ Same |")
            elif v0 is None:
                R.append(f"| `{k}` | _(not set)_ | `{v1}` | 🔄 New |")
            elif v1 is None:
                R.append(f"| `{k}` | `{v0}` | _(not set)_ | ❌ Missing |")
            else:
                R.append(f"| `{k}` | `{v0}` | `{v1}` | ⚠️ Changed |")
        R.append("")

    # Log event comparison
    R.append("---\n")
    R.append("## Log Event Comparison\n")
    for label, spans, events in [
        ("Original", spans_orig, ml_orig["events"]),
        ("Migrated", spans_migr, ml_migr["events"]),
    ]:
        R.append(f"### {label} event types (per trace)\n")
        by_trace = {}
        for e in events:
            by_trace.setdefault(e["traceId"][:12], []).append(e["event_name"])
        for i, s in enumerate(spans):
            sc = scenarios[i] if i < len(scenarios) else f"span_{i}"
            evts = by_trace.get(s["traceId"][:12], [])
            R.append(f"- **{sc}**: {', '.join(evts) if evts else '_(none)_'}")
        R.append("")

    # Metrics comparison
    R.append("---\n")
    R.append("## Metrics Comparison\n")
    m_orig = metric_summary(ml_orig["metrics"])
    m_migr = metric_summary(ml_migr["metrics"])
    all_keys = sorted(set(list(m_orig.keys()) + list(m_migr.keys())))
    if all_keys:
        R.append(
            "| Metric | Series | Original scope | Migrated scope | Original dims | Migrated dims | Status |"
        )
        R.append(
            "|--------|--------|----------------|----------------|---------------|---------------|--------|"
        )
        for k in all_keys:
            mo, mm = m_orig.get(k), m_migr.get(k)
            name = (mo or mm)["name"]
            disc_value = (mo or mm)["disc_value"]
            series_label = f"`{disc_value}`" if disc_value else "—"
            if mo and mm:
                d0 = ", ".join(sorted(mo["dp_attr_keys"]))
                d1 = ", ".join(sorted(mm["dp_attr_keys"]))
                status = "✅ Same" if d0 == d1 else "⚠️ Dims differ"
                R.append(
                    f"| `{name}` | {series_label} | `{mo['scope']}` | `{mm['scope']}` | {d0} | {d1} | {status} |"
                )
            elif mo:
                d0 = ", ".join(sorted(mo["dp_attr_keys"]))
                R.append(
                    f"| `{name}` | {series_label} | `{mo['scope']}` | _(missing)_ | {d0} | — | ❌ Missing |"
                )
            else:
                d1 = ", ".join(sorted(mm["dp_attr_keys"]))
                R.append(
                    f"| `{name}` | {series_label} | _(missing)_ | `{mm['scope']}` | — | {d1} | 🔄 New |"
                )
    else:
        R.append("No gen_ai metrics found.\n")
    R.append("")

    # Summary
    R.append("---\n")
    R.append("## Summary\n")
    R.append("Review the tables above and categorize each difference as:")
    R.append("- ✅ **Preserved** — same in both")
    R.append("- 🔄 **Intentionally changed** — expected per migration plan")
    R.append("- ❌ **Regression** — unexpected difference, needs fix")

    return "\n".join(R)


def main():
    parser = argparse.ArgumentParser(
        description="Compare OTLP telemetry between two instrumentation versions"
    )
    parser.add_argument(
        "--traces-file", required=True, help="Path to traces-export.json"
    )
    parser.add_argument(
        "--metrics-file",
        required=True,
        help="Path to metrics-export.json (contains metrics + logs)",
    )
    parser.add_argument(
        "--service-original",
        required=True,
        help="Service name for original run",
    )
    parser.add_argument(
        "--service-migrated",
        required=True,
        help="Service name for migrated run",
    )
    parser.add_argument(
        "--scenarios", required=True, help="Comma-separated scenario names"
    )
    parser.add_argument(
        "--realm",
        default="shw-playground",
        help="Splunk O11y realm for trace URLs",
    )
    parser.add_argument(
        "--output", default=None, help="Output markdown file (default: stdout)"
    )
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]

    # Parse data
    spans_orig = parse_spans(args.traces_file, args.service_original)
    spans_migr = parse_spans(args.traces_file, args.service_migrated)
    ml_orig = parse_metrics_and_logs(args.metrics_file, args.service_original)
    ml_migr = parse_metrics_and_logs(args.metrics_file, args.service_migrated)

    print(
        f"Parsed: {len(spans_orig)} original spans, {len(spans_migr)} migrated spans",
        file=sys.stderr,
    )
    print(
        f"Events: {len(ml_orig['events'])} original, {len(ml_migr['events'])} migrated",
        file=sys.stderr,
    )
    print(
        f"Metrics: {len(ml_orig['metrics'])} original, {len(ml_migr['metrics'])} migrated",
        file=sys.stderr,
    )

    report = generate_report(
        spans_orig,
        spans_migr,
        ml_orig,
        ml_migr,
        scenarios,
        args.service_original,
        args.service_migrated,
        args.realm,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()

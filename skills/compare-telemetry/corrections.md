# Corrections

Track issues and improvements for the compare-telemetry skill.

## v1.0 Lessons Learned

1. **Realm mismatch** — `SPLUNK_REALM` in collector config doesn't match browser org. Always ask user for correct realm or derive from a known-working URL.
2. **Collector log capture** — Start collector with `2>&1 | tee collector.log` so skill can read errors.
3. **Trace link format** — All gen_ai spans have the same name; flat lists are useless. Always use Scenario × PR table.
4. **Port conflicts** — Kill stale collector processes before restarting (check ports 4317, 8888).
5. **Log export 404** — `otlphttp/logs` to Splunk O11y returns 404; this is expected and not a trace/metric issue.

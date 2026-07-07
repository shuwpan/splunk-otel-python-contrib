# Example Report Structure

The comparison report follows this structure:

```markdown
# Telemetry Comparison: <service-original> vs <service-migrated>

## Trace Links (Splunk O11y)
| Scenario | Original | Migrated |
|----------|----------|----------|
| simple   | [link]   | [link]   |

## Pretty-Print Trace Trees
### Original
(scope, span name, filtered attributes, events per trace)

### Migrated
(same format)

## Span Attribute Comparison
### Scenario: <name>
| Attribute | Original | Migrated | Status |
(✅ Same / 🔄 New / ❌ Missing / ⚠️ Changed)

## Log Event Comparison
(event types per trace, both runs)

## Metrics Comparison
| Metric | Original scope | Migrated scope | Dims | Status |

## Summary
(categorization guidance)
```

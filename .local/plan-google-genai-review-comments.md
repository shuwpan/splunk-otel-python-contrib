# Google GenAI review comments

## Project description

Address review feedback on the Google GenAI instrumentation migration to the
util-genai lifecycle. The change keeps Google `generate_content` spans aligned
with current GenAI semantic conventions and prevents instrumentation-only
failures from affecting application behavior.

## Research

- Current OTel GenAI semantic conventions list `generate_content` as a
  predefined `gen_ai.operation.name` example for inference spans.
- Current OTel GenAI semantic conventions list `gcp.gen_ai` and
  `gcp.vertex_ai` as Google provider names. `google` is not listed as a
  well-known provider value.
- Existing util-genai `ToolCall` already owns `arguments` and `tool_result`
  fields for `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`, with
  emission controlled by the content-capture mode.

## Open questions

- None for the requested review fixes.

## Implementation plan for AI Coder

1. Add regression tests for operation/provider naming, response telemetry
   failure finalization, and tool-call telemetry isolation/content fields.
2. Patch Google GenAI invocation construction to use `generate_content` and
   semconv Google provider values.
3. Split response processing from LLM finalization so `stop_llm` or `fail_llm`
   always runs after a successful SDK response.
4. Patch tool-call wrapping so argument/result telemetry failures are swallowed
   and user tool execution semantics are preserved.
5. Update changelog and PR notes.
6. Run targeted tests and `make lint`, then stage changed files.

import asyncio
import os

from openai import OpenAI

# NOTE: OpenTelemetry Python Logs API is in beta
from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({"service.name": "openai-example-modified"})

# configure tracing
trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# configure logging
_logs.set_logger_provider(LoggerProvider(resource=resource))
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# configure metrics
metrics.set_meter_provider(
    MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(),
            ),
        ]
    )
)

# instrument OpenAI
OpenAIInstrumentor().instrument()


def get_clients():
    """Create sync and async OpenAI clients pointing to LM Studio."""
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        from openai import AsyncOpenAI
    except ImportError:  # pragma: no cover - optional convenience
        return client, None
    return client, AsyncOpenAI(base_url=base_url, api_key=api_key)


def weather_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]


def basic_chat_example(client: OpenAI, model: str, prompt: str) -> None:
    chat = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n=== basic_chat_example ===")
    print(chat.choices[0].message.content)


def streaming_chat_example(client: OpenAI, model: str, prompt: str) -> None:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    print("\n=== streaming_chat_example ===")
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


def non_stream_toolcall_example(client: OpenAI, model: str) -> None:
    tools = weather_tools()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in Portland?"}],
        tools=tools,
        tool_choice="auto",
    )
    print("\n=== non_stream_toolcall_example ===")
    print(response.choices[0].message)


def stream_toolcall_example(client: OpenAI, model: str) -> None:
    tools = weather_tools()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Weather in Portland?"}],
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    print("\n=== stream_toolcall_example ===")
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
        if delta.tool_calls:
            print(f"\n[tool_call delta] {delta.tool_calls}", flush=True)
    print()


def multi_turn_with_tool_example(client: OpenAI, model: str) -> None:
    tools = weather_tools()
    first = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Weather in Portland?"}],
        tools=tools,
        tool_choice="auto",
    )
    tool_calls = first.choices[0].message.tool_calls
    assert tool_calls is not None
    tool_call = tool_calls[0]
    messages = [
        {"role": "user", "content": "Weather in Portland?"},
        first.choices[0].message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": '{"location": "Portland", "unit": "celsius", "forecast": "rain"}',
        },
    ]
    followup = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    print("\n=== multi_turn_with_tool_example ===")
    print(followup.choices[0].message.content)


def streaming_multi_turn_with_tool_example(client: OpenAI, model: str) -> None:
    tools = weather_tools()
    first = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Weather in Portland?"}],
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    print("\n=== streaming_multi_turn_with_tool_example (step 1 stream) ===")
    tool_call_id = None
    for chunk in first:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
        if delta.tool_calls:
            tool_call_id = delta.tool_calls[0].id
            print(f"\n[tool_call delta] {delta.tool_calls}", flush=True)
    print()
    if not tool_call_id:
        print("No tool call produced; cannot continue multi-turn.")
        return
    messages = [
        {"role": "user", "content": "Weather in Portland?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Portland"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "get_weather",
            "content": '{"location": "Portland", "unit": "celsius", "forecast": "rain"}',
        },
    ]
    followup = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    print("\n=== streaming_multi_turn_with_tool_example (step 2 stream) ===")
    for chunk in followup:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


async def async_basic_chat_example(async_client, model: str, prompt: str) -> None:
    chat = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n=== async_basic_chat_example ===")
    print(chat.choices[0].message.content)


async def async_streaming_toolcall_example(async_client, model: str) -> None:
    tools = weather_tools()
    stream = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Weather in Portland?"}],
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    print("\n=== async_streaming_toolcall_example ===")
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
        if delta.tool_calls:
            print(f"\n[tool_call delta] {delta.tool_calls}", flush=True)
    print()


def embedding_example(client: OpenAI, model: str) -> None:
    embedding = client.embeddings.create(
        model=model,
        input="Hello from OpenTelemetry",
    )
    print("\n=== embedding_example ===")
    print(f"Embedding length: {len(embedding.data[0].embedding)}")


def batch_embedding_example(client: OpenAI, model: str) -> None:
    embedding = client.embeddings.create(
        model=model,
        input=["hello world", "otel telemetry"],
    )
    print("\n=== batch_embedding_example ===")
    print(f"Embeddings returned: {len(embedding.data)}")


async def async_embedding_example(async_client, model: str) -> None:
    embedding = await async_client.embeddings.create(
        model=model,
        input="Hello from OpenTelemetry (async)",
    )
    print("\n=== async_embedding_example ===")
    print(f"Embedding length: {len(embedding.data[0].embedding)}")


async def async_batch_embedding_example(async_client, model: str) -> None:
    embedding = await async_client.embeddings.create(
        model=model,
        input=["hello world async", "otel telemetry async"],
    )
    print("\n=== async_batch_embedding_example ===")
    print(f"Embeddings returned: {len(embedding.data)}")


def main():
    

    prompt = "Write a short poem on OpenTelemetry."
    model = os.getenv("CHAT_MODEL", "qwen/qwen3-vl-4b")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
    )
    client, async_client = get_clients()

    # Run whichever examples you want; comment out any you don't need.
    # basic_chat_example(client, model, prompt)
    # streaming_chat_example(client, model, prompt)
    # non_stream_toolcall_example(client, model)
    # stream_toolcall_example(client, model)
    multi_turn_with_tool_example(client, model)
    # streaming_multi_turn_with_tool_example(client, model)
    
    # embedding_example(client, embedding_model)
    # batch_embedding_example(client, embedding_model)
    
    # if async_client is None:
    #     print("AsyncOpenAI not installed; skipping async examples.")
    # else:
        # asyncio.run(async_basic_chat_example(async_client, model, prompt))
        # asyncio.run(async_streaming_toolcall_example(async_client, model))
    #     asyncio.run(async_embedding_example(async_client, embedding_model))
    #     asyncio.run(async_batch_embedding_example(async_client, embedding_model))


if __name__ == "__main__":
    main()

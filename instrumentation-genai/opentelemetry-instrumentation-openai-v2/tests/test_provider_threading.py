"""Test that telemetry handler receives the correct providers from instrumentor."""

import pytest
from unittest.mock import patch, MagicMock

from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

try:
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
except ImportError:
    from opentelemetry.sdk._logs.export import (
        InMemoryLogExporter as InMemoryLogRecordExporter,
    )

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class TestProviderThreading:
    """Tests to verify providers are properly threaded to genai-util handler."""

    def test_instrumentor_passes_providers_to_handler(self):
        """
        Verify that OpenAIInstrumentor passes the tracer/logger/meter providers
        to get_telemetry_handler, so spans/logs/metrics go through the user's
        configured pipeline instead of global defaults.
        """
        # Create custom providers (not the global ones)
        custom_tracer_provider = TracerProvider()
        span_exporter = InMemorySpanExporter()
        custom_tracer_provider.add_span_processor(
            SimpleSpanProcessor(span_exporter)
        )

        custom_logger_provider = LoggerProvider()
        log_exporter = InMemoryLogRecordExporter()
        custom_logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(log_exporter)
        )

        metric_reader = InMemoryMetricReader()
        custom_meter_provider = MeterProvider(metric_readers=[metric_reader])

        # Track calls to get_telemetry_handler
        handler_call_args = []

        # Import the original handler before patching
        from opentelemetry.util.genai.handler import (
            get_telemetry_handler as original_get_handler,
        )

        def tracking_get_handler(*args, **kwargs):
            handler_call_args.append(kwargs)
            # Call the original to avoid breaking functionality
            return original_get_handler(*args, **kwargs)

        # Patch get_telemetry_handler to track what providers are passed
        # Note: We patch in __init__ where it's now called during instrument()
        with patch(
            "opentelemetry.instrumentation.openai_v2.get_telemetry_handler"
        ) as mock_get_handler:
            # Make mock call through to real implementation
            mock_get_handler.side_effect = tracking_get_handler

            # Instrument with custom providers
            instrumentor = OpenAIInstrumentor()
            instrumentor.instrument(
                tracer_provider=custom_tracer_provider,
                logger_provider=custom_logger_provider,
                meter_provider=custom_meter_provider,
            )

            try:
                # The handler should have been called during instrument()
                # with our custom providers
                assert mock_get_handler.called, (
                    "get_telemetry_handler should be called during instrument()"
                )

                # Verify the providers were passed correctly
                # At least one call should have our providers
                provider_passed = False
                for call_kwargs in handler_call_args:
                    if (
                        call_kwargs.get("tracer_provider") is custom_tracer_provider
                        and call_kwargs.get("logger_provider") is custom_logger_provider
                        and call_kwargs.get("meter_provider") is custom_meter_provider
                    ):
                        provider_passed = True
                        break

                assert provider_passed, (
                    "get_telemetry_handler should be called with the custom providers "
                    f"passed to instrument(). Actual calls: {handler_call_args}"
                )

            finally:
                instrumentor.uninstrument()

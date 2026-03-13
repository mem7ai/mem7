"""OpenTelemetry integration for mem7."""

import json
from typing import Optional

from _mem7 import init_telemetry as _init_telemetry
from _mem7 import shutdown_telemetry as _shutdown_telemetry


def init_telemetry(
    *,
    otlp_endpoint: str = "http://localhost:4317",
    service_name: str = "mem7",
) -> None:
    """Initialize OpenTelemetry tracing with OTLP export.

    Sets up a tracing subscriber that exports spans via OTLP/gRPC.
    Call ``shutdown_telemetry()`` before process exit to flush pending spans.

    Args:
        otlp_endpoint: OTLP collector endpoint (default ``http://localhost:4317``).
        service_name: Service name reported in traces (default ``mem7``).
    """
    config = {
        "otlp_endpoint": otlp_endpoint,
        "service_name": service_name,
    }
    _init_telemetry(json.dumps(config))


def shutdown_telemetry() -> None:
    """Flush pending spans and shut down the OTLP exporter."""
    _shutdown_telemetry()

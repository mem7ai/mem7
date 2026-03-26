"""mem7 - High-performance AI memory engine with Rust core."""

import importlib.metadata

from mem7.memory import AsyncMemory, Memory

__all__ = ["Memory", "AsyncMemory", "init_telemetry", "shutdown_telemetry"]

try:
    __version__ = importlib.metadata.version("mem7")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.3.3"


def init_telemetry(**kwargs):
    """Initialize OpenTelemetry tracing. See ``mem7.telemetry`` for full docs."""
    from mem7.telemetry import init_telemetry as _init

    return _init(**kwargs)


def shutdown_telemetry():
    """Flush pending spans and shut down the OTLP exporter."""
    from mem7.telemetry import shutdown_telemetry as _shutdown

    return _shutdown()

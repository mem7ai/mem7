use mem7_config::TelemetryConfig;
use mem7_error::Result;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{
    EnvFilter, Layer, Registry, layer::SubscriberExt, util::SubscriberInitExt,
};

static PROVIDER: std::sync::OnceLock<SdkTracerProvider> = std::sync::OnceLock::new();

/// Initialize OpenTelemetry tracing with OTLP export.
///
/// Sets up a global `tracing` subscriber that:
/// 1. Emits human-readable logs to stderr (controlled by `RUST_LOG`).
/// 2. Exports spans as OpenTelemetry traces via OTLP/gRPC.
///
/// Call [`shutdown`] before process exit to flush pending spans.
pub fn init(config: &TelemetryConfig) -> Result<()> {
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.otlp_endpoint)
        .build()
        .map_err(|e| mem7_error::Mem7Error::Config(format!("OTLP exporter: {e}")))?;

    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .build();

    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build();

    let tracer = provider.tracer("mem7");

    PROVIDER
        .set(provider)
        .map_err(|_| mem7_error::Mem7Error::Config("telemetry already initialized".into()))?;

    let otel_layer = OpenTelemetryLayer::new(tracer);

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_level(true)
        .with_filter(EnvFilter::from_default_env());

    Registry::default().with(fmt_layer).with(otel_layer).init();

    Ok(())
}

/// Flush pending spans and shut down the OTLP exporter.
pub fn shutdown() {
    if let Some(provider) = PROVIDER.get()
        && let Err(e) = provider.shutdown()
    {
        tracing::warn!(error = %e, "failed to shutdown telemetry provider");
    }
}

use mem7_config::TelemetryConfig;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn init_telemetry(config_json: Option<String>) -> Result<()> {
    let config: TelemetryConfig = match config_json {
        Some(json) => serde_json::from_str(&json).map_err(|e| {
            Error::new(Status::InvalidArg, format!("Invalid telemetry config: {e}"))
        })?,
        None => TelemetryConfig::default(),
    };

    mem7_telemetry::init(&config).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to init telemetry: {e}"),
        )
    })
}

#[napi]
pub fn shutdown_telemetry() {
    mem7_telemetry::shutdown();
}

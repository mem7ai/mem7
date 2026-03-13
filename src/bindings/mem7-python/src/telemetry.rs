use mem7_config::TelemetryConfig;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (config_json=None))]
pub fn init_telemetry(config_json: Option<&str>) -> PyResult<()> {
    let config: TelemetryConfig = match config_json {
        Some(json) => serde_json::from_str(json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid telemetry config: {e}")))?,
        None => TelemetryConfig::default(),
    };

    mem7_telemetry::init(&config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to init telemetry: {e}")))
}

#[pyfunction]
pub fn shutdown_telemetry() {
    mem7_telemetry::shutdown();
}

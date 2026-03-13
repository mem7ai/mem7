mod async_engine;
mod engine;
mod telemetry;
mod types;

use pyo3::prelude::*;

#[pymodule]
fn _mem7(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyMemoryEngine>()?;
    m.add_class::<async_engine::PyAsyncMemoryEngine>()?;
    m.add_class::<types::PyMemoryItem>()?;
    m.add_class::<types::PyMemoryActionResult>()?;
    m.add_class::<types::PyGraphRelation>()?;
    m.add_class::<types::PyAddResult>()?;
    m.add_class::<types::PySearchResult>()?;
    m.add_class::<types::PyMemoryEvent>()?;
    m.add_function(wrap_pyfunction!(telemetry::init_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::shutdown_telemetry, m)?)?;
    Ok(())
}

mod engine;
mod types;

use pyo3::prelude::*;

#[pymodule]
fn _mem7(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyMemoryEngine>()?;
    Ok(())
}

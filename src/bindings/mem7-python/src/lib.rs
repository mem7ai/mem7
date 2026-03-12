mod engine;
mod types;

use pyo3::prelude::*;

#[pymodule]
fn _mem7(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyMemoryEngine>()?;
    m.add_class::<types::PyMemoryItem>()?;
    m.add_class::<types::PyMemoryActionResult>()?;
    m.add_class::<types::PyAddResult>()?;
    m.add_class::<types::PySearchResult>()?;
    m.add_class::<types::PyMemoryEvent>()?;
    Ok(())
}

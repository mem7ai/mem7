use mem7_error::Mem7Error;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert a Mem7Error into a Python RuntimeError.
pub fn to_py_err(e: Mem7Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

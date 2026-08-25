use pyo3::prelude::*;
use queryweave_core::{Document, QueryWeaveEngine, SearchRequest};

#[pyclass(name = "Engine")]
struct PyEngine { inner: QueryWeaveEngine }

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> Self { Self { inner: QueryWeaveEngine::new() } }

    fn upsert_json(&self, payload: &str) -> PyResult<usize> {
        let documents: Vec<Document> = serde_json::from_str(payload)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(self.inner.upsert(documents))
    }

    fn search_json(&self, payload: &str) -> PyResult<String> {
        let request: SearchRequest = serde_json::from_str(payload)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        serde_json::to_string(&self.inner.search(request))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn stats_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.stats())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn reset(&self) { self.inner.reset(); }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}

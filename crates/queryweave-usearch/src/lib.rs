#![forbid(unsafe_code)]
//! HNSW/quantized ANN backend powered by USearch.

use queryweave_core::{Document, VectorIndex};
use std::collections::HashSet;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    F32,
    I8,
}

pub struct USearchHnswIndex {
    index: Option<Index>,
    dimensions: usize,
    quantization: Quantization,
    connectivity: usize,
    expansion_add: usize,
    expansion_search: usize,
}

impl Default for USearchHnswIndex {
    fn default() -> Self {
        Self::f32()
    }
}

impl USearchHnswIndex {
    pub fn f32() -> Self {
        Self::new(Quantization::F32)
    }

    pub fn i8() -> Self {
        Self::new(Quantization::I8)
    }

    pub fn new(quantization: Quantization) -> Self {
        Self {
            index: None,
            dimensions: 0,
            quantization,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }

    pub fn with_tuning(
        mut self,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
    ) -> Self {
        self.connectivity = connectivity.max(4);
        self.expansion_add = expansion_add.max(16);
        self.expansion_search = expansion_search.max(8);
        self
    }

    fn scalar_kind(&self) -> ScalarKind {
        match self.quantization {
            Quantization::F32 => ScalarKind::F32,
            Quantization::I8 => ScalarKind::I8,
        }
    }
}

impl VectorIndex for USearchHnswIndex {
    fn name(&self) -> &'static str {
        match self.quantization {
            Quantization::F32 => "usearch-hnsw-f32",
            Quantization::I8 => "usearch-hnsw-i8",
        }
    }

    fn rebuild(&mut self, documents: &[Document]) {
        self.index = None;
        self.dimensions = documents
            .iter()
            .find_map(|document| document.dense.as_ref().map(Vec::len))
            .unwrap_or(0);
        if self.dimensions == 0 || documents.is_empty() {
            return;
        }

        let options = IndexOptions {
            dimensions: self.dimensions,
            metric: MetricKind::Cos,
            quantization: self.scalar_kind(),
            connectivity: self.connectivity,
            expansion_add: self.expansion_add,
            expansion_search: self.expansion_search,
            ..Default::default()
        };
        let Ok(index) = Index::new(&options) else {
            return;
        };
        if index.reserve(documents.len()).is_err() {
            return;
        }

        for (document_index, document) in documents.iter().enumerate() {
            let Some(vector) = &document.dense else {
                continue;
            };
            if vector.len() != self.dimensions || index.add(document_index as u64, vector).is_err() {
                self.index = None;
                return;
            }
        }
        self.index = Some(index);
    }

    fn search(
        &self,
        query: &[f32],
        eligible: &HashSet<usize>,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let Some(index) = &self.index else {
            return Vec::new();
        };
        if query.len() != self.dimensions || eligible.is_empty() {
            return Vec::new();
        }

        let allowed = eligible.clone();
        let Ok(matches) = index.filtered_search(query, limit, move |key| {
            allowed.contains(&(key as usize))
        }) else {
            return Vec::new();
        };

        matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(key, distance)| (*key as usize, 1.0 - *distance))
            .collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use queryweave_core::Metadata;

    fn document(id: &str, vector: Vec<f32>) -> Document {
        Document {
            id: id.into(),
            text: id.into(),
            source: "test".into(),
            metadata: Metadata::new(),
            dense: Some(vector),
            sparse: None,
        }
    }

    #[test]
    fn hnsw_returns_nearest_vector_and_respects_filter() {
        let mut index = USearchHnswIndex::f32();
        index.rebuild(&[
            document("a", vec![1.0, 0.0, 0.0]),
            document("b", vec![0.0, 1.0, 0.0]),
        ]);
        let eligible = HashSet::from([0usize]);
        let hits = index.search(&[1.0, 0.0, 0.0], &eligible, 2);
        assert_eq!(hits.first().map(|hit| hit.0), Some(0));
    }

    #[test]
    fn quantized_i8_backend_builds() {
        let mut index = USearchHnswIndex::i8();
        index.rebuild(&[document("a", vec![1.0, 0.0, 0.0])]);
        assert_eq!(index.dimensions(), 3);
        assert_eq!(index.name(), "usearch-hnsw-i8");
    }
}

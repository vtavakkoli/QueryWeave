#![forbid(unsafe_code)]
//! Tantivy-backed BM25 lexical retrieval for QueryWeave.

use queryweave_core::{rare_ratio_from_df, simple_tokenize, Document, LexicalRetriever};
use std::collections::{HashMap, HashSet};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::Value;
use tantivy::schema::{Field, Schema, TantivyDocument, STORED, TEXT};
use tantivy::{doc, Index, IndexReader};

#[derive(Default)]
pub struct TantivyLexicalIndex {
    index: Option<Index>,
    reader: Option<IndexReader>,
    body_field: Option<Field>,
    doc_index_field: Option<Field>,
    df: HashMap<String, usize>,
    document_count: usize,
}

impl TantivyLexicalIndex {
    fn clear_runtime(&mut self) {
        self.index = None;
        self.reader = None;
        self.body_field = None;
        self.doc_index_field = None;
    }

    fn rebuild_df(&mut self, documents: &[Document]) {
        self.df.clear();
        self.document_count = documents.len();
        for document in documents {
            let unique: HashSet<String> = simple_tokenize(&document.text).into_iter().collect();
            for term in unique {
                *self.df.entry(term).or_insert(0) += 1;
            }
        }
    }
}

impl LexicalRetriever for TantivyLexicalIndex {
    fn name(&self) -> &'static str {
        "tantivy-bm25"
    }

    fn rebuild(&mut self, documents: &[Document]) {
        self.clear_runtime();
        self.rebuild_df(documents);
        if documents.is_empty() {
            return;
        }

        let mut schema_builder = Schema::builder();
        let body_field = schema_builder.add_text_field("body", TEXT);
        let doc_index_field = schema_builder.add_u64_field("doc_index", STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let Ok(mut writer) = index.writer(50_000_000) else {
            return;
        };

        for (document_index, document) in documents.iter().enumerate() {
            let _ = writer.add_document(doc!(
                body_field => document.text.as_str(),
                doc_index_field => document_index as u64,
            ));
        }
        if writer.commit().is_err() {
            return;
        }
        let Ok(reader) = index.reader() else {
            return;
        };

        self.body_field = Some(body_field);
        self.doc_index_field = Some(doc_index_field);
        self.reader = Some(reader);
        self.index = Some(index);
    }

    fn search(
        &self,
        query: &str,
        eligible: &HashSet<usize>,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let (Some(index), Some(reader), Some(body_field), Some(doc_index_field)) = (
            &self.index,
            &self.reader,
            self.body_field,
            self.doc_index_field,
        ) else {
            return Vec::new();
        };
        if eligible.is_empty() || limit == 0 {
            return Vec::new();
        }

        let parser = QueryParser::for_index(index, vec![body_field]);
        let Ok(parsed) = parser.parse_query(query) else {
            return Vec::new();
        };
        let searcher = reader.searcher();
        let num_docs = searcher.num_docs() as usize;
        if num_docs == 0 {
            return Vec::new();
        }
        let candidate_limit = (limit.max(1) * 8).min(num_docs).max(1);
        let collector = TopDocs::with_limit(candidate_limit).order_by_score();
        let Ok(top_docs) = searcher.search(&parsed, &collector) else {
            return Vec::new();
        };

        let mut output = Vec::with_capacity(limit);
        for (score, address) in top_docs {
            let Ok(stored) = searcher.doc::<TantivyDocument>(address) else {
                continue;
            };
            let Some(document_index) = stored
                .get_first(doc_index_field)
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
            else {
                continue;
            };
            if eligible.contains(&document_index) {
                output.push((document_index, score));
                if output.len() == limit {
                    break;
                }
            }
        }
        output
    }

    fn rare_ratio(&self, query: &str) -> f32 {
        rare_ratio_from_df(query, &self.df, self.document_count)
    }

    fn term_count(&self) -> usize {
        self.df.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use queryweave_core::Metadata;

    fn document(id: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            text: text.into(),
            source: "test".into(),
            metadata: Metadata::new(),
            dense: None,
            sparse: None,
        }
    }

    #[test]
    fn tantivy_bm25_finds_exact_terms() {
        let mut index = TantivyLexicalIndex::default();
        index.rebuild(&[
            document("a", "CVE-2026-12345 security advisory"),
            document("b", "coffee opening hours"),
        ]);
        let eligible = HashSet::from([0usize, 1usize]);
        let hits = index.search("CVE-2026-12345", &eligible, 2);
        assert_eq!(hits.first().map(|hit| hit.0), Some(0));
        assert_eq!(index.name(), "tantivy-bm25");
    }

    #[test]
    fn zero_limit_returns_no_hits() {
        let mut index = TantivyLexicalIndex::default();
        index.rebuild(&[document("a", "rust hybrid search")]);
        let eligible = HashSet::from([0usize]);
        assert!(index.search("rust", &eligible, 0).is_empty());
    }
}

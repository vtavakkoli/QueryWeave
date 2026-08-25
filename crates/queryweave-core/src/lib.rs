#![forbid(unsafe_code)]
//! QueryWeave core: adaptive lexical + sparse + dense retrieval with explainable fusion.
//!
//! The default implementation is deliberately dependency-light and deterministic. Production
//! deployments can replace the exact dense backend, encoders, and reranker through the public
//! traits without changing the query/fusion contract.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::RwLock;
use std::time::Instant;

pub type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn normalized(self) -> Self {
        let mut aggregated = BTreeMap::<u32, f32>::new();
        for (index, value) in self.indices.into_iter().zip(self.values) {
            *aggregated.entry(index).or_insert(0.0) += value;
        }
        Self {
            indices: aggregated.keys().copied().collect(),
            values: aggregated.values().copied().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub metadata: Metadata,
    #[serde(default)]
    pub dense: Option<Vec<f32>>,
    #[serde(default)]
    pub sparse: Option<SparseVector>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    #[default]
    Auto,
    Lexical,
    Hybrid,
    Deep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default)]
    pub dense: Option<Vec<f32>>,
    #[serde(default)]
    pub sparse: Option<SparseVector>,
    #[serde(default)]
    pub filter: Metadata,
    #[serde(default = "default_true")]
    pub explain: bool,
}

fn default_limit() -> usize {
    10
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FusionWeights {
    pub lexical: f32,
    pub sparse: f32,
    pub dense: f32,
}

impl FusionWeights {
    fn normalize(mut self) -> Self {
        let total = (self.lexical + self.sparse + self.dense).max(f32::EPSILON);
        self.lexical /= total;
        self.sparse /= total;
        self.dense /= total;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryFeatures {
    pub token_count: usize,
    pub numeric_ratio: f32,
    pub identifier_ratio: f32,
    pub rare_ratio: f32,
    pub lexical_margin: f32,
    pub retriever_disagreement: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComponentScores {
    pub lexical: f32,
    pub sparse: f32,
    pub dense: f32,
    pub rerank: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub route: String,
    pub weights: FusionWeights,
    pub features: QueryFeatures,
    pub early_exit: bool,
    pub reranked: bool,
    pub reranker: String,
    pub candidate_pool: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub text: String,
    pub source: String,
    pub metadata: Metadata,
    pub score: f32,
    pub components: ComponentScores,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<Explanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub route: String,
    pub elapsed_ms: f64,
    pub indexed_documents: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineStats {
    pub documents: usize,
    pub dense_dimensions: usize,
    pub lexical_terms: usize,
    pub sparse_dimensions_observed: usize,
}

#[derive(Debug, Clone)]
struct ScoredDoc {
    doc_idx: usize,
    score: f32,
}

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
}

pub trait SparseEncoder: Send + Sync {
    fn encode_sparse(&self, text: &str) -> SparseVector;
}

pub trait VectorIndex: Send + Sync {
    fn rebuild(&mut self, documents: &[Document]);
    fn search(
        &self,
        query: &[f32],
        eligible: &HashSet<usize>,
        limit: usize,
    ) -> Vec<(usize, f32)>;
    fn dimensions(&self) -> usize;
}

pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;
    fn rerank(&self, query: &str, hits: &mut [SearchHit]);
}

#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dimensions: usize,
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self { dimensions: 128 }
    }
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(8),
        }
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimensions];
        for token in tokenize(text) {
            let hash = stable_hash(token.as_bytes());
            let index = (hash as usize) % self.dimensions;
            let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }
        l2_normalize(&mut vector);
        vector
    }
}

#[derive(Debug, Clone, Default)]
pub struct HashSparseEncoder;

impl SparseEncoder for HashSparseEncoder {
    fn encode_sparse(&self, text: &str) -> SparseVector {
        let mut counts = BTreeMap::<u32, f32>::new();
        for token in tokenize(text) {
            let index = (stable_hash(token.as_bytes()) % 65_536) as u32;
            *counts.entry(index).or_insert(0.0) += 1.0;
        }
        SparseVector {
            indices: counts.keys().copied().collect(),
            values: counts.values().map(|value| (1.0 + *value).ln()).collect(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExactVectorIndex {
    vectors: Vec<Option<Vec<f32>>>,
    dimensions: usize,
}

impl VectorIndex for ExactVectorIndex {
    fn rebuild(&mut self, documents: &[Document]) {
        self.dimensions = documents
            .iter()
            .find_map(|document| document.dense.as_ref().map(Vec::len))
            .unwrap_or(0);
        self.vectors = documents
            .iter()
            .map(|document| document.dense.clone())
            .collect();
    }

    fn search(
        &self,
        query: &[f32],
        eligible: &HashSet<usize>,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(index, _)| eligible.contains(index))
            .filter_map(|(index, value)| value.as_ref().map(|vector| (index, cosine(query, vector))))
            .filter(|(_, score)| score.is_finite())
            .collect();
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.truncate(limit);
        scored
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[derive(Debug, Clone, Default)]
struct LexicalIndex {
    doc_terms: Vec<HashMap<String, usize>>,
    doc_lengths: Vec<usize>,
    df: HashMap<String, usize>,
    avg_len: f32,
}

impl LexicalIndex {
    fn rebuild(&mut self, documents: &[Document]) {
        self.doc_terms.clear();
        self.doc_lengths.clear();
        self.df.clear();

        for document in documents {
            let tokens = tokenize(&document.text);
            let mut term_frequency = HashMap::new();
            let mut unique = BTreeSet::new();
            for token in &tokens {
                *term_frequency.entry(token.clone()).or_insert(0usize) += 1;
                unique.insert(token.clone());
            }
            for term in unique {
                *self.df.entry(term).or_insert(0) += 1;
            }
            self.doc_lengths.push(tokens.len());
            self.doc_terms.push(term_frequency);
        }

        self.avg_len = if self.doc_lengths.is_empty() {
            0.0
        } else {
            self.doc_lengths.iter().sum::<usize>() as f32 / self.doc_lengths.len() as f32
        };
    }

    fn search(&self, query: &str, eligible: &HashSet<usize>, limit: usize) -> Vec<ScoredDoc> {
        let query_terms = tokenize(query);
        let document_count = self.doc_terms.len() as f32;
        let mut results = Vec::new();

        for (index, terms) in self.doc_terms.iter().enumerate() {
            if !eligible.contains(&index) {
                continue;
            }

            let mut score = 0.0;
            let document_length = self.doc_lengths[index] as f32;
            for term in &query_terms {
                let term_frequency = *terms.get(term).unwrap_or(&0) as f32;
                if term_frequency == 0.0 {
                    continue;
                }
                let document_frequency = *self.df.get(term).unwrap_or(&0) as f32;
                let idf = (1.0
                    + ((document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)))
                    .ln();
                let k1 = 1.2;
                let b = 0.75;
                let length_norm = if self.avg_len > 0.0 {
                    document_length / self.avg_len
                } else {
                    1.0
                };
                score += idf * (term_frequency * (k1 + 1.0))
                    / (term_frequency + k1 * (1.0 - b + b * length_norm));
            }

            if score > 0.0 {
                results.push(ScoredDoc {
                    doc_idx: index,
                    score,
                });
            }
        }

        results.sort_by(|left, right| right.score.total_cmp(&left.score));
        results.truncate(limit);
        results
    }

    fn rare_ratio(&self, query: &str) -> f32 {
        let query_terms = tokenize(query);
        if query_terms.is_empty() || self.doc_terms.is_empty() {
            return 0.0;
        }
        let threshold = ((self.doc_terms.len() as f32) * 0.05).ceil() as usize;
        query_terms
            .iter()
            .filter(|term| self.df.get(*term).copied().unwrap_or(0) <= threshold.max(1))
            .count() as f32
            / query_terms.len() as f32
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinLateInteractionReranker;

impl Reranker for BuiltinLateInteractionReranker {
    fn name(&self) -> &'static str {
        "token-maxsim-proxy"
    }

    fn rerank(&self, query: &str, hits: &mut [SearchHit]) {
        let query_tokens = tokenize(query);
        let query_set: HashSet<&str> = query_tokens.iter().map(String::as_str).collect();

        for hit in hits.iter_mut() {
            let document_tokens = tokenize(&hit.text);
            let document_set: HashSet<&str> =
                document_tokens.iter().map(String::as_str).collect();
            let overlap = if query_set.is_empty() {
                0.0
            } else {
                query_set
                    .iter()
                    .filter(|token| document_set.contains(**token))
                    .count() as f32
                    / query_set.len() as f32
            };
            hit.components.rerank = overlap;
            hit.score = 0.82 * hit.score + 0.18 * overlap;
        }
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    }
}

#[derive(Default)]
struct State {
    documents: Vec<Document>,
    lexical: LexicalIndex,
    dense: ExactVectorIndex,
    sparse_dimensions_observed: usize,
}

impl State {
    fn rebuild(&mut self) {
        self.lexical.rebuild(&self.documents);
        self.dense.rebuild(&self.documents);
        let mut dimensions = HashSet::new();
        for document in &self.documents {
            if let Some(sparse) = &document.sparse {
                dimensions.extend(sparse.indices.iter().copied());
            }
        }
        self.sparse_dimensions_observed = dimensions.len();
    }
}

pub struct QueryWeaveEngine {
    state: RwLock<State>,
    embedder: Box<dyn Embedder>,
    sparse_encoder: Box<dyn SparseEncoder>,
}

impl Default for QueryWeaveEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryWeaveEngine {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(State::default()),
            embedder: Box::new(HashEmbedder::default()),
            sparse_encoder: Box::new(HashSparseEncoder),
        }
    }

    pub fn with_encoders(
        embedder: Box<dyn Embedder>,
        sparse_encoder: Box<dyn SparseEncoder>,
    ) -> Self {
        Self {
            state: RwLock::new(State::default()),
            embedder,
            sparse_encoder,
        }
    }

    pub fn upsert(&self, mut documents: Vec<Document>) -> usize {
        let mut state = self.state.write().expect("queryweave state poisoned");
        for document in &mut documents {
            if document.dense.is_none() {
                document.dense = Some(self.embedder.embed(&document.text));
            }
            if document.sparse.is_none() {
                document.sparse = Some(self.sparse_encoder.encode_sparse(&document.text));
            }
            if let Some(sparse) = document.sparse.take() {
                document.sparse = Some(sparse.normalized());
            }
        }

        let count = documents.len();
        for document in documents {
            if let Some(position) = state
                .documents
                .iter()
                .position(|existing| existing.id == document.id)
            {
                state.documents[position] = document;
            } else {
                state.documents.push(document);
            }
        }
        state.rebuild();
        count
    }

    pub fn reset(&self) {
        let mut state = self.state.write().expect("queryweave state poisoned");
        *state = State::default();
    }

    pub fn stats(&self) -> EngineStats {
        let state = self.state.read().expect("queryweave state poisoned");
        EngineStats {
            documents: state.documents.len(),
            dense_dimensions: state.dense.dimensions(),
            lexical_terms: state.lexical.df.len(),
            sparse_dimensions_observed: state.sparse_dimensions_observed,
        }
    }

    pub fn search(&self, request: SearchRequest) -> SearchResponse {
        self.search_with_reranker(request, None)
    }

    pub fn search_with_reranker(
        &self,
        mut request: SearchRequest,
        reranker: Option<&dyn Reranker>,
    ) -> SearchResponse {
        let started = Instant::now();
        let state = self.state.read().expect("queryweave state poisoned");
        if state.documents.is_empty() {
            return SearchResponse {
                hits: Vec::new(),
                route: "empty".into(),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                indexed_documents: 0,
            };
        }

        request.limit = request.limit.clamp(1, 100);
        let candidate_limit = (request.limit * 6)
            .min(state.documents.len())
            .max(request.limit);
        let eligible: HashSet<usize> = state
            .documents
            .iter()
            .enumerate()
            .filter(|(_, document)| matches_filter(document, &request.filter))
            .map(|(index, _)| index)
            .collect();

        let lexical = state
            .lexical
            .search(&request.query, &eligible, candidate_limit);
        let query_dense = request
            .dense
            .clone()
            .unwrap_or_else(|| self.embedder.embed(&request.query));
        let query_sparse = request
            .sparse
            .clone()
            .unwrap_or_else(|| self.sparse_encoder.encode_sparse(&request.query))
            .normalized();
        let dense: Vec<ScoredDoc> = state
            .dense
            .search(&query_dense, &eligible, candidate_limit)
            .into_iter()
            .map(|(doc_idx, score)| ScoredDoc { doc_idx, score })
            .collect();
        let sparse = sparse_search(
            &state.documents,
            &query_sparse,
            &eligible,
            candidate_limit,
        );

        let retriever_disagreement = disagreement(&lexical, &dense, request.limit);
        let lexical_margin = score_margin(&lexical);
        let features = query_features(
            &request.query,
            state.lexical.rare_ratio(&request.query),
            lexical_margin,
            retriever_disagreement,
        );
        let weights = adaptive_weights(&features);
        let early_exit = request.mode == SearchMode::Auto
            && features.identifier_ratio >= 0.34
            && features.lexical_margin >= 0.55
            && !lexical.is_empty();

        let (route, mut fused) = match request.mode {
            SearchMode::Lexical => ("lexical", fuse_single(&lexical, "lexical")),
            SearchMode::Hybrid => (
                "hybrid",
                fuse_adaptive(&lexical, &sparse, &dense, weights),
            ),
            SearchMode::Deep => ("deep", fuse_adaptive(&lexical, &sparse, &dense, weights)),
            SearchMode::Auto if early_exit => {
                ("lexical_early_exit", fuse_single(&lexical, "lexical"))
            }
            SearchMode::Auto if retriever_disagreement > 0.62 || lexical_margin < 0.12 => {
                ("deep", fuse_adaptive(&lexical, &sparse, &dense, weights))
            }
            SearchMode::Auto => (
                "hybrid",
                fuse_adaptive(&lexical, &sparse, &dense, weights),
            ),
        };

        fused.truncate(candidate_limit);
        let mut hits: Vec<SearchHit> = fused
            .into_iter()
            .map(|(index, score, components)| {
                let document = &state.documents[index];
                SearchHit {
                    id: document.id.clone(),
                    text: document.text.clone(),
                    source: document.source.clone(),
                    metadata: document.metadata.clone(),
                    score,
                    components,
                    explanation: if request.explain {
                        Some(Explanation {
                            route: route.into(),
                            weights,
                            features: features.clone(),
                            early_exit,
                            reranked: false,
                            reranker: "none".into(),
                            candidate_pool: candidate_limit,
                        })
                    } else {
                        None
                    },
                }
            })
            .collect();

        let should_rerank = route == "deep" && !hits.is_empty();
        if should_rerank {
            if let Some(custom) = reranker {
                custom.rerank(&request.query, &mut hits);
                mark_reranked(&mut hits, custom.name());
            } else {
                let builtin = BuiltinLateInteractionReranker;
                builtin.rerank(&request.query, &mut hits);
                mark_reranked(&mut hits, builtin.name());
            }
        }

        hits.truncate(request.limit);
        SearchResponse {
            hits,
            route: route.into(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            indexed_documents: state.documents.len(),
        }
    }
}

fn mark_reranked(hits: &mut [SearchHit], name: &str) {
    for hit in hits {
        if let Some(explanation) = &mut hit.explanation {
            explanation.reranked = true;
            explanation.reranker = name.into();
        }
    }
}

fn query_features(
    query: &str,
    rare_ratio: f32,
    lexical_margin: f32,
    disagreement: f32,
) -> QueryFeatures {
    let words: Vec<&str> = query.split_whitespace().collect();
    let token_count = words.len();
    if words.is_empty() {
        return QueryFeatures::default();
    }

    let numeric = words
        .iter()
        .filter(|word| word.chars().any(|character| character.is_ascii_digit()))
        .count();
    let identifiers = words
        .iter()
        .filter(|word| {
            let has_digit = word.chars().any(|character| character.is_ascii_digit());
            let has_delimiter = word
                .chars()
                .any(|character| matches!(character, '-' | '_' | ':' | '/' | '#' | '.'));
            has_digit && has_delimiter
        })
        .count();

    QueryFeatures {
        token_count,
        numeric_ratio: numeric as f32 / words.len() as f32,
        identifier_ratio: identifiers as f32 / words.len() as f32,
        rare_ratio,
        lexical_margin,
        retriever_disagreement: disagreement,
    }
}

fn adaptive_weights(features: &QueryFeatures) -> FusionWeights {
    let mut weights = if features.identifier_ratio >= 0.25 || features.numeric_ratio >= 0.5 {
        FusionWeights {
            lexical: 0.62,
            sparse: 0.28,
            dense: 0.10,
        }
    } else if features.token_count >= 9 {
        FusionWeights {
            lexical: 0.18,
            sparse: 0.27,
            dense: 0.55,
        }
    } else {
        FusionWeights {
            lexical: 0.30,
            sparse: 0.30,
            dense: 0.40,
        }
    };

    if features.rare_ratio > 0.5 {
        weights.lexical += 0.08;
        weights.sparse += 0.05;
        weights.dense -= 0.08;
    }
    if features.retriever_disagreement > 0.65 {
        weights.sparse += 0.06;
        weights.dense += 0.04;
    }
    if features.lexical_margin > 0.55 {
        weights.lexical += 0.12;
        weights.dense -= 0.06;
    }

    weights.lexical = weights.lexical.max(0.02);
    weights.sparse = weights.sparse.max(0.02);
    weights.dense = weights.dense.max(0.02);
    weights.normalize()
}

fn fuse_adaptive(
    lexical: &[ScoredDoc],
    sparse: &[ScoredDoc],
    dense: &[ScoredDoc],
    weights: FusionWeights,
) -> Vec<(usize, f32, ComponentScores)> {
    let lexical_scores = normalize_scores(lexical);
    let sparse_scores = normalize_scores(sparse);
    let dense_scores = normalize_scores(dense);
    let mut candidates = HashSet::new();
    candidates.extend(lexical_scores.keys().copied());
    candidates.extend(sparse_scores.keys().copied());
    candidates.extend(dense_scores.keys().copied());

    let mut output = Vec::new();
    for index in candidates {
        let components = ComponentScores {
            lexical: *lexical_scores.get(&index).unwrap_or(&0.0),
            sparse: *sparse_scores.get(&index).unwrap_or(&0.0),
            dense: *dense_scores.get(&index).unwrap_or(&0.0),
            rerank: 0.0,
        };
        let score = components.lexical * weights.lexical
            + components.sparse * weights.sparse
            + components.dense * weights.dense;
        output.push((index, score, components));
    }
    output.sort_by(|left, right| right.1.total_cmp(&left.1));
    output
}

fn fuse_single(scores: &[ScoredDoc], kind: &str) -> Vec<(usize, f32, ComponentScores)> {
    normalize_scores(scores)
        .into_iter()
        .map(|(index, score)| {
            let mut components = ComponentScores::default();
            match kind {
                "lexical" => components.lexical = score,
                "sparse" => components.sparse = score,
                _ => components.dense = score,
            }
            (index, score, components)
        })
        .collect()
}

fn normalize_scores(scores: &[ScoredDoc]) -> HashMap<usize, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }
    let min = scores
        .iter()
        .map(|scored| scored.score)
        .fold(f32::INFINITY, f32::min);
    let max = scores
        .iter()
        .map(|scored| scored.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(1e-6);
    scores
        .iter()
        .map(|scored| {
            let normalized = if scores.len() == 1 {
                1.0
            } else {
                (scored.score - min) / span
            };
            (scored.doc_idx, normalized)
        })
        .collect()
}

pub fn reciprocal_rank_fusion(lists: &[Vec<(String, f32)>], k: f32) -> Vec<(String, f32)> {
    let mut scores = HashMap::<String, f32>::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut output: Vec<_> = scores.into_iter().collect();
    output.sort_by(|left, right| right.1.total_cmp(&left.1));
    output
}

fn sparse_search(
    documents: &[Document],
    query: &SparseVector,
    eligible: &HashSet<usize>,
    limit: usize,
) -> Vec<ScoredDoc> {
    let mut output = Vec::new();
    for (index, document) in documents.iter().enumerate() {
        if !eligible.contains(&index) {
            continue;
        }
        if let Some(sparse) = &document.sparse {
            let score = sparse_dot(query, sparse);
            if score > 0.0 {
                output.push(ScoredDoc {
                    doc_idx: index,
                    score,
                });
            }
        }
    }
    output.sort_by(|left, right| right.score.total_cmp(&left.score));
    output.truncate(limit);
    output
}

fn sparse_dot(left: &SparseVector, right: &SparseVector) -> f32 {
    let (mut left_index, mut right_index, mut score) = (0usize, 0usize, 0.0f32);
    while left_index < left.indices.len() && right_index < right.indices.len() {
        match left.indices[left_index].cmp(&right.indices[right_index]) {
            std::cmp::Ordering::Equal => {
                score += left.values.get(left_index).copied().unwrap_or(0.0)
                    * right.values.get(right_index).copied().unwrap_or(0.0);
                left_index += 1;
                right_index += 1;
            }
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
        }
    }
    score
}

fn score_margin(scores: &[ScoredDoc]) -> f32 {
    match scores {
        [] => 0.0,
        [_] => 1.0,
        _ => ((scores[0].score - scores[1].score) / scores[0].score.abs().max(1e-6))
            .clamp(0.0, 1.0),
    }
}

fn disagreement(left: &[ScoredDoc], right: &[ScoredDoc], k: usize) -> f32 {
    let left_top: HashSet<usize> = left.iter().take(k).map(|item| item.doc_idx).collect();
    let right_top: HashSet<usize> = right.iter().take(k).map(|item| item.doc_idx).collect();
    if left_top.is_empty() && right_top.is_empty() {
        return 0.0;
    }
    let intersection = left_top.intersection(&right_top).count() as f32;
    let union = left_top.union(&right_top).count() as f32;
    1.0 - intersection / union.max(1.0)
}

fn matches_filter(document: &Document, filter: &Metadata) -> bool {
    filter.iter().all(|(key, value)| {
        if key == "source" {
            &document.source == value
        } else {
            document.metadata.get(key) == Some(value)
        }
    })
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric() && character != '_' && character != '-')
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn exact_identifier_prefers_lexical_route() {
        let engine = QueryWeaveEngine::new();
        engine.upsert(vec![
            document("a", "CVE-2026-12345 security advisory"),
            document("b", "general security guidance"),
        ]);
        let response = engine.search(SearchRequest {
            query: "CVE-2026-12345".into(),
            limit: 2,
            mode: SearchMode::Auto,
            dense: None,
            sparse: None,
            filter: Metadata::new(),
            explain: true,
        });
        assert_eq!(response.hits[0].id, "a");
        assert!(response.route.contains("lexical") || response.route == "hybrid");
    }

    #[test]
    fn semantic_query_uses_multiple_signals() {
        let engine = QueryWeaveEngine::new();
        engine.upsert(vec![
            document(
                "pump",
                "hydraulic pump temperature failure and predictive maintenance",
            ),
            document("coffee", "coffee shop opening hours"),
        ]);
        let response = engine.search(SearchRequest {
            query: "ways to prevent industrial pump failure caused by excessive heat".into(),
            limit: 2,
            mode: SearchMode::Auto,
            dense: None,
            sparse: None,
            filter: Metadata::new(),
            explain: true,
        });
        assert_eq!(response.hits[0].id, "pump");
        assert!(response.hits[0].components.lexical >= 0.0);
    }

    #[test]
    fn metadata_filter_is_enforced() {
        let engine = QueryWeaveEngine::new();
        let mut english = document("a", "rust search engine");
        english.metadata.insert("lang".into(), "en".into());
        let mut german = document("b", "rust suchmaschine");
        german.metadata.insert("lang".into(), "de".into());
        engine.upsert(vec![english, german]);

        let mut filter = Metadata::new();
        filter.insert("lang".into(), "de".into());
        let response = engine.search(SearchRequest {
            query: "rust".into(),
            limit: 10,
            mode: SearchMode::Hybrid,
            dense: None,
            sparse: None,
            filter,
            explain: true,
        });
        assert_eq!(response.hits.len(), 1);
        assert_eq!(response.hits[0].id, "b");
    }

    #[test]
    fn sparse_normalization_merges_duplicate_dimensions() {
        let vector = SparseVector {
            indices: vec![4, 2, 4],
            values: vec![0.5, 1.0, 0.75],
        }
        .normalized();
        assert_eq!(vector.indices, vec![2, 4]);
        assert_eq!(vector.values, vec![1.0, 1.25]);
    }

    #[test]
    fn rrf_rewards_consensus() {
        let lists = vec![
            vec![("a".into(), 1.0), ("b".into(), 0.9)],
            vec![("a".into(), 0.8), ("c".into(), 0.7)],
        ];
        let fused = reciprocal_rank_fusion(&lists, 60.0);
        assert_eq!(fused[0].0, "a");
    }
}

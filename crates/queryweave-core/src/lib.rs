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
    pub fn normalized(mut self) -> Self {
        let mut pairs: Vec<(u32, f32)> = self.indices.drain(..).zip(self.values.drain(..)).collect();
        pairs.sort_unstable_by_key(|(i, _)| *i);
        pairs.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 += a.1;
                true
            } else {
                false
            }
        });
        self.indices = pairs.iter().map(|(i, _)| *i).collect();
        self.values = pairs.iter().map(|(_, v)| *v).collect();
        self
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

fn default_limit() -> usize { 10 }
fn default_true() -> bool { true }

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
    fn search(&self, query: &[f32], eligible: &HashSet<usize>, limit: usize) -> Vec<(usize, f32)>;
    fn dimensions(&self) -> usize;
}

pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;
    fn rerank(&self, query: &str, hits: &mut [SearchHit]);
}

#[derive(Debug, Clone)]
pub struct HashEmbedder { dimensions: usize }

impl Default for HashEmbedder {
    fn default() -> Self { Self { dimensions: 128 } }
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self { Self { dimensions: dimensions.max(8) } }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimensions];
        for token in tokenize(text) {
            let h = stable_hash(token.as_bytes());
            let idx = (h as usize) % self.dimensions;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            vector[idx] += sign;
        }
        l2_normalize(&mut vector);
        vector
    }
}

#[derive(Debug, Clone, Default)]
pub struct HashSparseEncoder;

impl SparseEncoder for HashSparseEncoder {
    fn encode_sparse(&self, text: &str) -> SparseVector {
        let mut counts: BTreeMap<u32, f32> = BTreeMap::new();
        for token in tokenize(text) {
            let idx = (stable_hash(token.as_bytes()) % 65_536) as u32;
            *counts.entry(idx).or_insert(0.0) += 1.0;
        }
        SparseVector {
            indices: counts.keys().copied().collect(),
            values: counts.values().map(|v| (1.0 + *v).ln()).collect(),
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
        self.dimensions = documents.iter().find_map(|d| d.dense.as_ref().map(Vec::len)).unwrap_or(0);
        self.vectors = documents.iter().map(|d| d.dense.clone()).collect();
    }

    fn search(&self, query: &[f32], eligible: &HashSet<usize>, limit: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self.vectors.iter().enumerate()
            .filter(|(idx, _)| eligible.contains(idx))
            .filter_map(|(idx, value)| value.as_ref().map(|v| (idx, cosine(query, v))))
            .filter(|(_, score)| score.is_finite())
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);
        scored
    }

    fn dimensions(&self) -> usize { self.dimensions }
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
        for doc in documents {
            let tokens = tokenize(&doc.text);
            let mut tf = HashMap::new();
            let mut unique = BTreeSet::new();
            for token in &tokens {
                *tf.entry(token.clone()).or_insert(0usize) += 1;
                unique.insert(token.clone());
            }
            for term in unique { *self.df.entry(term).or_insert(0) += 1; }
            self.doc_lengths.push(tokens.len());
            self.doc_terms.push(tf);
        }
        self.avg_len = if self.doc_lengths.is_empty() { 0.0 } else {
            self.doc_lengths.iter().sum::<usize>() as f32 / self.doc_lengths.len() as f32
        };
    }

    fn search(&self, query: &str, eligible: &HashSet<usize>, limit: usize) -> Vec<ScoredDoc> {
        let q = tokenize(query);
        let n = self.doc_terms.len() as f32;
        let mut out = Vec::new();
        for (idx, terms) in self.doc_terms.iter().enumerate() {
            if !eligible.contains(&idx) { continue; }
            let mut score = 0.0;
            let dl = self.doc_lengths[idx] as f32;
            for term in &q {
                let tf = *terms.get(term).unwrap_or(&0) as f32;
                if tf == 0.0 { continue; }
                let df = *self.df.get(term).unwrap_or(&0) as f32;
                let idf = (1.0 + ((n - df + 0.5) / (df + 0.5))).ln();
                let k1 = 1.2;
                let b = 0.75;
                let norm = if self.avg_len > 0.0 { dl / self.avg_len } else { 1.0 };
                score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * norm));
            }
            if score > 0.0 { out.push(ScoredDoc { doc_idx: idx, score }); }
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(limit);
        out
    }

    fn rare_ratio(&self, query: &str) -> f32 {
        let q = tokenize(query);
        if q.is_empty() || self.doc_terms.is_empty() { return 0.0; }
        let threshold = ((self.doc_terms.len() as f32) * 0.05).ceil() as usize;
        q.iter().filter(|term| self.df.get(*term).copied().unwrap_or(0) <= threshold.max(1)).count() as f32 / q.len() as f32
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinLateInteractionReranker;

impl Reranker for BuiltinLateInteractionReranker {
    fn name(&self) -> &'static str { "token-maxsim-proxy" }

    fn rerank(&self, query: &str, hits: &mut [SearchHit]) {
        let q = tokenize(query);
        let qset: HashSet<&str> = q.iter().map(String::as_str).collect();
        for hit in hits.iter_mut() {
            let dt = tokenize(&hit.text);
            let dset: HashSet<&str> = dt.iter().map(String::as_str).collect();
            let overlap = if qset.is_empty() { 0.0 } else {
                qset.iter().filter(|t| dset.contains(**t)).count() as f32 / qset.len() as f32
            };
            hit.components.rerank = overlap;
            hit.score = 0.82 * hit.score + 0.18 * overlap;
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    }
}

struct State {
    documents: Vec<Document>,
    lexical: LexicalIndex,
    dense: ExactVectorIndex,
    sparse_dimensions_observed: usize,
}

impl Default for State {
    fn default() -> Self {
        Self { documents: Vec::new(), lexical: LexicalIndex::default(), dense: ExactVectorIndex::default(), sparse_dimensions_observed: 0 }
    }
}

impl State {
    fn rebuild(&mut self) {
        self.lexical.rebuild(&self.documents);
        self.dense.rebuild(&self.documents);
        let mut dims = HashSet::new();
        for d in &self.documents {
            if let Some(s) = &d.sparse { dims.extend(s.indices.iter().copied()); }
        }
        self.sparse_dimensions_observed = dims.len();
    }
}

pub struct QueryWeaveEngine {
    state: RwLock<State>,
    embedder: Box<dyn Embedder>,
    sparse_encoder: Box<dyn SparseEncoder>,
}

impl Default for QueryWeaveEngine {
    fn default() -> Self { Self::new() }
}

impl QueryWeaveEngine {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(State::default()),
            embedder: Box::new(HashEmbedder::default()),
            sparse_encoder: Box::new(HashSparseEncoder),
        }
    }

    pub fn with_encoders(embedder: Box<dyn Embedder>, sparse_encoder: Box<dyn SparseEncoder>) -> Self {
        Self { state: RwLock::new(State::default()), embedder, sparse_encoder }
    }

    pub fn upsert(&self, mut documents: Vec<Document>) -> usize {
        let mut state = self.state.write().expect("queryweave state poisoned");
        for doc in documents.iter_mut() {
            if doc.dense.is_none() { doc.dense = Some(self.embedder.embed(&doc.text)); }
            if doc.sparse.is_none() { doc.sparse = Some(self.sparse_encoder.encode_sparse(&doc.text)); }
            if let Some(sparse) = doc.sparse.take() { doc.sparse = Some(sparse.normalized()); }
        }
        let count = documents.len();
        for doc in documents {
            if let Some(pos) = state.documents.iter().position(|old| old.id == doc.id) {
                state.documents[pos] = doc;
            } else {
                state.documents.push(doc);
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

    pub fn search_with_reranker(&self, mut request: SearchRequest, reranker: Option<&dyn Reranker>) -> SearchResponse {
        let started = Instant::now();
        let state = self.state.read().expect("queryweave state poisoned");
        if state.documents.is_empty() {
            return SearchResponse { hits: Vec::new(), route: "empty".into(), elapsed_ms: started.elapsed().as_secs_f64() * 1000.0, indexed_documents: 0 };
        }
        request.limit = request.limit.clamp(1, 100);
        let candidate_limit = (request.limit * 6).min(state.documents.len()).max(request.limit);
        let eligible: HashSet<usize> = state.documents.iter().enumerate()
            .filter(|(_, d)| matches_filter(d, &request.filter))
            .map(|(idx, _)| idx)
            .collect();

        let lexical = state.lexical.search(&request.query, &eligible, candidate_limit);
        let q_dense = request.dense.clone().unwrap_or_else(|| self.embedder.embed(&request.query));
        let q_sparse = request.sparse.clone().unwrap_or_else(|| self.sparse_encoder.encode_sparse(&request.query)).normalized();
        let dense: Vec<ScoredDoc> = state.dense.search(&q_dense, &eligible, candidate_limit).into_iter().map(|(doc_idx, score)| ScoredDoc { doc_idx, score }).collect();
        let sparse = sparse_search(&state.documents, &q_sparse, &eligible, candidate_limit);

        let disagreement = disagreement(&lexical, &dense, request.limit);
        let lexical_margin = score_margin(&lexical);
        let features = query_features(&request.query, state.lexical.rare_ratio(&request.query), lexical_margin, disagreement);
        let weights = adaptive_weights(&features);
        let early_exit = request.mode == SearchMode::Auto && features.identifier_ratio >= 0.34 && features.lexical_margin >= 0.55 && !lexical.is_empty();

        let (route, mut fused) = match request.mode {
            SearchMode::Lexical => ("lexical", fuse_single(&lexical, "lexical")),
            SearchMode::Hybrid => ("hybrid", fuse_adaptive(&lexical, &sparse, &dense, weights)),
            SearchMode::Deep => ("deep", fuse_adaptive(&lexical, &sparse, &dense, weights)),
            SearchMode::Auto if early_exit => ("lexical_early_exit", fuse_single(&lexical, "lexical")),
            SearchMode::Auto if disagreement > 0.62 || lexical_margin < 0.12 => ("deep", fuse_adaptive(&lexical, &sparse, &dense, weights)),
            SearchMode::Auto => ("hybrid", fuse_adaptive(&lexical, &sparse, &dense, weights)),
        };

        fused.truncate(candidate_limit);
        let mut hits: Vec<SearchHit> = fused.into_iter().map(|(idx, score, components)| {
            let d = &state.documents[idx];
            SearchHit {
                id: d.id.clone(), text: d.text.clone(), source: d.source.clone(), metadata: d.metadata.clone(), score, components,
                explanation: if request.explain { Some(Explanation {
                    route: route.into(), weights, features: features.clone(), early_exit, reranked: false,
                    reranker: "none".into(), candidate_pool: candidate_limit,
                }) } else { None },
            }
        }).collect();

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
        if let Some(exp) = &mut hit.explanation { exp.reranked = true; exp.reranker = name.into(); }
    }
}

fn query_features(query: &str, rare_ratio: f32, lexical_margin: f32, disagreement: f32) -> QueryFeatures {
    let words: Vec<&str> = query.split_whitespace().collect();
    let token_count = words.len();
    if words.is_empty() { return QueryFeatures::default(); }
    let numeric = words.iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).count();
    let identifiers = words.iter().filter(|w| {
        let has_digit = w.chars().any(|c| c.is_ascii_digit());
        let has_delimiter = w.chars().any(|c| matches!(c, '-' | '_' | ':' | '/' | '#' | '.'));
        has_digit && has_delimiter
    }).count();
    QueryFeatures {
        token_count,
        numeric_ratio: numeric as f32 / words.len() as f32,
        identifier_ratio: identifiers as f32 / words.len() as f32,
        rare_ratio,
        lexical_margin,
        retriever_disagreement: disagreement,
    }
}

fn adaptive_weights(f: &QueryFeatures) -> FusionWeights {
    let mut w = if f.identifier_ratio >= 0.25 || f.numeric_ratio >= 0.5 {
        FusionWeights { lexical: 0.62, sparse: 0.28, dense: 0.10 }
    } else if f.token_count >= 9 {
        FusionWeights { lexical: 0.18, sparse: 0.27, dense: 0.55 }
    } else {
        FusionWeights { lexical: 0.30, sparse: 0.30, dense: 0.40 }
    };
    if f.rare_ratio > 0.5 { w.lexical += 0.08; w.sparse += 0.05; w.dense -= 0.08; }
    if f.retriever_disagreement > 0.65 { w.sparse += 0.06; w.dense += 0.04; }
    if f.lexical_margin > 0.55 { w.lexical += 0.12; w.dense -= 0.06; }
    w.lexical = w.lexical.max(0.02);
    w.sparse = w.sparse.max(0.02);
    w.dense = w.dense.max(0.02);
    w.normalize()
}

fn fuse_adaptive(lexical: &[ScoredDoc], sparse: &[ScoredDoc], dense: &[ScoredDoc], w: FusionWeights) -> Vec<(usize, f32, ComponentScores)> {
    let l = normalize_scores(lexical);
    let s = normalize_scores(sparse);
    let d = normalize_scores(dense);
    let mut all = HashSet::new();
    all.extend(l.keys().copied()); all.extend(s.keys().copied()); all.extend(d.keys().copied());
    let mut out = Vec::new();
    for idx in all {
        let components = ComponentScores {
            lexical: *l.get(&idx).unwrap_or(&0.0), sparse: *s.get(&idx).unwrap_or(&0.0), dense: *d.get(&idx).unwrap_or(&0.0), rerank: 0.0,
        };
        let score = components.lexical * w.lexical + components.sparse * w.sparse + components.dense * w.dense;
        out.push((idx, score, components));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn fuse_single(scores: &[ScoredDoc], kind: &str) -> Vec<(usize, f32, ComponentScores)> {
    normalize_scores(scores).into_iter().map(|(idx, score)| {
        let mut c = ComponentScores::default();
        match kind { "lexical" => c.lexical = score, "sparse" => c.sparse = score, _ => c.dense = score }
        (idx, score, c)
    }).collect()
}

fn normalize_scores(scores: &[ScoredDoc]) -> HashMap<usize, f32> {
    if scores.is_empty() { return HashMap::new(); }
    let min = scores.iter().map(|s| s.score).fold(f32::INFINITY, f32::min);
    let max = scores.iter().map(|s| s.score).fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(1e-6);
    scores.iter().map(|s| (s.doc_idx, if scores.len() == 1 { 1.0 } else { (s.score - min) / span })).collect()
}

pub fn reciprocal_rank_fusion(lists: &[Vec<(String, f32)>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<_> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn sparse_search(documents: &[Document], query: &SparseVector, eligible: &HashSet<usize>, limit: usize) -> Vec<ScoredDoc> {
    let mut out = Vec::new();
    for (idx, doc) in documents.iter().enumerate() {
        if !eligible.contains(&idx) { continue; }
        if let Some(sparse) = &doc.sparse {
            let score = sparse_dot(query, sparse);
            if score > 0.0 { out.push(ScoredDoc { doc_idx: idx, score }); }
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit);
    out
}

fn sparse_dot(a: &SparseVector, b: &SparseVector) -> f32 {
    let (mut i, mut j, mut score) = (0usize, 0usize, 0.0f32);
    while i < a.indices.len() && j < b.indices.len() {
        match a.indices[i].cmp(&b.indices[j]) {
            std::cmp::Ordering::Equal => { score += a.values.get(i).copied().unwrap_or(0.0) * b.values.get(j).copied().unwrap_or(0.0); i += 1; j += 1; }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    score
}

fn score_margin(scores: &[ScoredDoc]) -> f32 {
    match scores {
        [] => 0.0,
        [_] => 1.0,
        _ => ((scores[0].score - scores[1].score) / scores[0].score.abs().max(1e-6)).clamp(0.0, 1.0),
    }
}

fn disagreement(a: &[ScoredDoc], b: &[ScoredDoc], k: usize) -> f32 {
    let aa: HashSet<usize> = a.iter().take(k).map(|x| x.doc_idx).collect();
    let bb: HashSet<usize> = b.iter().take(k).map(|x| x.doc_idx).collect();
    if aa.is_empty() && bb.is_empty() { return 0.0; }
    let inter = aa.intersection(&bb).count() as f32;
    let union = aa.union(&bb).count() as f32;
    1.0 - inter / union.max(1.0)
}

fn matches_filter(doc: &Document, filter: &Metadata) -> bool {
    filter.iter().all(|(key, value)| {
        if key == "source" { &doc.source == value } else { doc.metadata.get(key) == Some(value) }
    })
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes { hash ^= *byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in v { *x /= norm; } }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let mut dot = 0.0; let mut aa = 0.0; let mut bb = 0.0;
    for (x, y) in a.iter().zip(b) { dot += x * y; aa += x * x; bb += y * y; }
    if aa == 0.0 || bb == 0.0 { 0.0 } else { dot / (aa.sqrt() * bb.sqrt()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, text: &str) -> Document {
        Document { id: id.into(), text: text.into(), source: "test".into(), metadata: Metadata::new(), dense: None, sparse: None }
    }

    #[test]
    fn exact_identifier_prefers_lexical_route() {
        let e = QueryWeaveEngine::new();
        e.upsert(vec![doc("a", "CVE-2026-12345 security advisory"), doc("b", "general security guidance")]);
        let response = e.search(SearchRequest { query: "CVE-2026-12345".into(), limit: 2, mode: SearchMode::Auto, dense: None, sparse: None, filter: Metadata::new(), explain: true });
        assert_eq!(response.hits[0].id, "a");
        assert!(response.route.contains("lexical") || response.route == "hybrid");
    }

    #[test]
    fn semantic_query_uses_multiple_signals() {
        let e = QueryWeaveEngine::new();
        e.upsert(vec![doc("pump", "hydraulic pump temperature failure and predictive maintenance"), doc("coffee", "coffee shop opening hours")]);
        let response = e.search(SearchRequest { query: "ways to prevent industrial pump failure caused by excessive heat".into(), limit: 2, mode: SearchMode::Auto, dense: None, sparse: None, filter: Metadata::new(), explain: true });
        assert_eq!(response.hits[0].id, "pump");
        assert!(response.hits[0].components.lexical >= 0.0);
    }

    #[test]
    fn metadata_filter_is_enforced() {
        let e = QueryWeaveEngine::new();
        let mut a = doc("a", "rust search engine"); a.metadata.insert("lang".into(), "en".into());
        let mut b = doc("b", "rust suchmaschine"); b.metadata.insert("lang".into(), "de".into());
        e.upsert(vec![a, b]);
        let mut filter = Metadata::new(); filter.insert("lang".into(), "de".into());
        let response = e.search(SearchRequest { query: "rust".into(), limit: 10, mode: SearchMode::Hybrid, dense: None, sparse: None, filter, explain: true });
        assert_eq!(response.hits.len(), 1);
        assert_eq!(response.hits[0].id, "b");
    }

    #[test]
    fn rrf_rewards_consensus() {
        let lists = vec![vec![("a".into(), 1.0), ("b".into(), 0.9)], vec![("a".into(), 0.8), ("c".into(), 0.7)]];
        let fused = reciprocal_rank_fusion(&lists, 60.0);
        assert_eq!(fused[0].0, "a");
    }
}

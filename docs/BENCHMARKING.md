# Benchmark methodology

QueryWeave should be evaluated as both an **algorithm** and an **engine**.

## 1. Algorithm ablations

Use identical documents, chunks, dense vectors and sparse vectors. Compare:

1. lexical only;
2. dense only;
3. sparse only;
4. RRF;
5. fixed normalized fusion;
6. AQF;
7. AQF + always-on reranker;
8. AQF + confidence-triggered reranker.

This shows where quality gains actually come from.

## 2. Engine comparison

Run QueryWeave, Qdrant, Elasticsearch, Vespa, LanceDB and Meilisearch on identical hardware where possible. Pin container/package versions in published runs. Do not compare one engine using BGE/SPLADE against another using different embeddings.

The companion `Hybrid-Search` repository integrates QueryWeave as a third backend beside Qdrant and Elasticsearch and reuses its existing FastEmbed BGE/SPLADE vectors.

## 3. Datasets

Recommended starting set:

- BEIR NFCorpus;
- SciFact;
- FiQA;
- TREC-COVID;
- ArguAna;
- DBPedia;
- MS MARCO for scale/throughput work.

## 4. Report

Report at minimum:

- nDCG@10;
- MRR@10;
- Recall@100;
- p50/p95/p99 latency;
- QPS;
- ingestion docs/s;
- resident memory;
- persisted index size;
- CPU/query;
- reranker invocation percentage.

Plot quality vs latency and quality vs compute. A system is Pareto-better only when no compared system improves one objective without worsening another.

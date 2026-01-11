# arxiv-sorter

Rank arXiv papers by relevance to your research interests using LLM pairwise comparisons.

## The problem

You search arXiv for "attention mechanisms" and get 500 papers. Which ones matter to you? Keyword matching won't help. You care about transformer efficiency, not attention in cognitive science or computer vision.

## The approach

Instead of asking an LLM to score each paper 1-10 (unreliable, poorly calibrated), we use pairwise comparisons: "Which paper is more relevant to X?" This is a much easier question for the model to answer correctly.

The naive approach requires O(n²) comparisons. We use comparison-based sorting instead, which gets us down to O(n log n). The comparisons are batched and parallelized using [autobatcher](https://github.com/doublewordai/autobatcher) and [parfold](https://github.com/doublewordai/parfold), so a 100-paper ranking completes in a few minutes rather than hours.

## Installation

```bash
uv add arxiv-sorter
```

You'll need:
- An OpenAI-compatible API endpoint (set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`)
- A parquet file of arXiv papers (set `ARXIV_PARQUET_PATH`)

## Usage

### Rank papers by relevance

```bash
arxiv-sorter rank "attention mechanisms" -i "transformer efficiency for long sequences"
```

The `-i` flag takes your research interest as a string, or a path to a file containing a longer description. More context helps the model make better comparisons.

Options:
- `-s cs.LG` filters by arXiv subject
- `-n 200` searches more papers (default 100)
- `--algorithm mergesort` uses mergesort instead of quicksort
- `--trace trace.json` outputs every comparison for debugging
- `--no-filter` skips the relevance pre-filter

### Quick search without ranking

```bash
arxiv-sorter search "transformer" -s cs.CL -n 20
```

Returns papers matching your query, sorted by date. Useful for checking what's in your dataset before running a full ranking.

### Cluster papers into themes

```bash
arxiv-sorter cluster ranked_papers.txt -o clusters.txt
```

Takes a list of papers (the output format from `rank`) and groups them into thematic clusters using a parallel fold. Each merge step asks the LLM to combine clusters with similar themes. Useful for making sense of a large set of relevant papers.

## How it works

1. **Search**: DuckDB queries a parquet file of arXiv metadata (title, abstract, subject, date).

2. **Filter** (optional): Each paper is checked for relevance with a simple yes/no prompt. This cuts down the candidate set before the expensive ranking step.

3. **Rank**: Papers are sorted using quicksort or mergesort, where each comparison is an LLM call asking "which paper is more relevant to [interest]?" Results are cached so repeated comparisons (common in sorting) don't waste API calls.

4. **Cluster** (separate command): Papers are grouped using a parallel fold. Start with single-paper clusters, then repeatedly merge pairs of cluster groups, asking the LLM to combine thematically similar clusters.

The batching layer ([autobatcher](https://github.com/doublewordai/autobatcher)) collects concurrent requests and submits them as OpenAI batch jobs, which are cheaper and have higher rate limits than real-time requests. The parallel primitives ([parfold](https://github.com/doublewordai/parfold)) handle the concurrency.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `ARXIV_PARQUET_PATH` | `./arxiv.parquet` | Path to arXiv data |
| `ARXIV_SORTER_MODEL` | `gpt-4o-mini` | Model for comparisons |
| `ARXIV_SORTER_BATCH_SIZE` | `100` | Requests per batch |

"""Core ranking logic for arXiv paper sorter."""

import re
from dataclasses import dataclass
from typing import Callable

import duckdb
from tqdm import tqdm

from autobatcher import BatchOpenAI
from parfold import quicksort, mergesort, filter as pfilter, fold

from arxiv_sorter import config


@dataclass
class ComparisonTrace:
    """Trace data for a single pairwise comparison."""
    paper_a: dict
    paper_b: dict
    prompt: str
    response: str
    winner: str  # "A" or "B"


TraceCallback = Callable[[ComparisonTrace], None]


def create_batch_client() -> BatchOpenAI:
    """Create the batching OpenAI client."""
    if not config.API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return BatchOpenAI(
        base_url=config.API_BASE_URL,
        api_key=config.API_KEY,
        batch_size=config.BATCH_SIZE,
        batch_window_seconds=config.BATCH_WINDOW_SECONDS,
        poll_interval_seconds=config.POLL_INTERVAL_SECONDS,
        completion_window="24h",
    )


def search_papers(
    query: str,
    subject: str | None = None,
    limit: int = 20,
    parquet_path: str | None = None,
) -> list[dict]:
    """Search arXiv papers using DuckDB."""
    con = duckdb.connect()
    path = parquet_path or config.PARQUET_PATH

    conditions = [f"(title ILIKE '%{query}%' OR abstract ILIKE '%{query}%')"]
    if subject:
        conditions.append(f"primary_subject LIKE '%{subject}%'")

    sql = f"""
        SELECT arxiv_id, title, abstract, primary_subject, submission_date
        FROM '{path}'
        WHERE {" AND ".join(conditions)}
        ORDER BY submission_date DESC
        LIMIT {limit}
    """

    result = con.execute(sql).fetchall()
    columns = ["arxiv_id", "title", "abstract", "primary_subject", "submission_date"]
    return [dict(zip(columns, row)) for row in result]


def format_paper(paper: dict) -> str:
    """Format a single paper for LLM display."""
    abstract = paper.get("abstract", "")[:300]
    date = paper.get("submission_date", "unknown")
    return f"[{paper['arxiv_id']}] ({date}) {paper['title']}\n{abstract}..."


def make_relevance_prompt(user_interest: str, paper: str) -> str:
    """Create prompt for relevance filtering."""
    return f"""Is this paper relevant to: {user_interest}

<paper>
{paper}
</paper>

Reply with ONLY "YES" or "NO"."""


def make_compare_prompt(user_interest: str, paper_a: str, paper_b: str) -> str:
    """Create prompt for pairwise comparison."""
    return f"""Which paper is MORE relevant to: {user_interest}

<paper_a>
{paper_a}
</paper_a>

<paper_b>
{paper_b}
</paper_b>

Reply with ONLY "A" or "B"."""


def extract_relevance(response: str) -> bool:
    """Parse relevance response."""
    response = response.strip().upper()
    return "YES" in response and "NO" not in response


def extract_winner(response: str) -> str:
    """Parse comparison response to get winner."""
    response = response.strip().upper()
    if "A" in response and "B" not in response:
        return "A"
    return "B"  # Default to B on ambiguity


async def filter_papers(
    papers: list[dict],
    user_interest: str,
    client: BatchOpenAI,
    show_progress: bool = True,
) -> list[dict]:
    """
    Filter papers by relevance using parallel LLM evaluation.

    Args:
        papers: List of paper dicts
        user_interest: What the user cares about
        client: BatchOpenAI client
        show_progress: Whether to show progress bar

    Returns:
        Papers that are relevant to the user's interest
    """
    if not papers:
        return []

    pbar = tqdm(
        total=len(papers),
        desc="Filtering",
        unit="paper",
        disable=not show_progress,
    )

    async def is_relevant(paper: dict) -> bool:
        formatted = format_paper(paper)
        prompt = make_relevance_prompt(user_interest, formatted)

        response = await client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        pbar.update(1)
        return extract_relevance(response.choices[0].message.content or "")

    # Use parallel filter
    relevant = await pfilter(papers, is_relevant)
    pbar.close()

    return relevant


async def rank_papers(
    papers: list[dict],
    user_interest: str,
    client: BatchOpenAI,
    algorithm: str = "quicksort",
    show_progress: bool = True,
    on_trace: TraceCallback | None = None,
) -> list[dict]:
    """
    Rank papers by relevance using parallel sorting with pairwise LLM comparisons.

    Args:
        papers: List of paper dicts
        user_interest: What the user cares about
        client: BatchOpenAI client for efficient batched requests
        algorithm: "quicksort" or "mergesort"
        show_progress: Whether to show progress bar
        on_trace: Optional callback for each comparison (for debugging)

    Returns:
        Papers sorted by relevance (most relevant first)
    """
    if len(papers) <= 1:
        return papers

    # Format papers for comparison - keep (formatted_str, paper_dict) pairs
    formatted = [(format_paper(p), p) for p in papers]

    # Progress tracking
    import math
    n = len(papers)
    estimated_comparisons = int(n * math.log2(n + 1)) if n > 1 else 0
    pbar = tqdm(
        total=estimated_comparisons,
        desc="Ranking",
        unit="cmp",
        disable=not show_progress,
    )

    # Cache for comparison results (symmetric: if A>B then B<A)
    cache: dict[tuple[str, str], int] = {}

    # Build the comparison function with caching and tracing
    async def compare(a: tuple[str, dict], b: tuple[str, dict]) -> int:
        key = (a[1]["arxiv_id"], b[1]["arxiv_id"])
        rev_key = (b[1]["arxiv_id"], a[1]["arxiv_id"])

        # Check cache
        if key in cache:
            pbar.update(1)
            return cache[key]
        if rev_key in cache:
            pbar.update(1)
            return -cache[rev_key]

        prompt = make_compare_prompt(user_interest, a[0], b[0])

        response = await client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        response_text = response.choices[0].message.content or ""
        winner = extract_winner(response_text)

        # Cache result
        result = -1 if winner == "A" else 1
        cache[key] = result

        pbar.update(1)

        # Trace if callback provided
        if on_trace:
            on_trace(ComparisonTrace(
                paper_a=a[1],
                paper_b=b[1],
                prompt=prompt,
                response=response_text,
                winner=winner,
            ))

        return result

    # Sort using chosen algorithm
    sort_fn = quicksort if algorithm == "quicksort" else mergesort
    sorted_pairs = await sort_fn(formatted, compare)

    pbar.close()

    # Extract papers in sorted order
    return [pair[1] for pair in sorted_pairs]


# =============================================================================
# Clustering via parallel fold
# =============================================================================


def parse_paper_list(text: str) -> list[str]:
    """
    Parse output.txt format into list of paper entries.

    Format: "N. [arxiv_id] Title"
    """
    papers = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match pattern: optional number, then [arxiv_id] Title
        match = re.match(r"^\d*\.?\s*(\[.+)", line)
        if match:
            papers.append(match.group(1))
    return papers


def lookup_abstracts(
    paper_entries: list[str],
    parquet_path: str | None = None,
) -> list[dict]:
    """
    Look up abstracts for papers from parquet file.

    Returns list of dicts with arxiv_id, title, abstract.
    """
    # Extract arxiv IDs
    arxiv_ids = []
    for entry in paper_entries:
        match = re.match(r"\[([^\]]+)\]", entry)
        if match:
            arxiv_ids.append(match.group(1))

    if not arxiv_ids:
        return [{"entry": e, "abstract": ""} for e in paper_entries]

    con = duckdb.connect()
    path = parquet_path or config.PARQUET_PATH

    # Query for abstracts
    ids_str = ", ".join(f"'{id}'" for id in arxiv_ids)
    sql = f"""
        SELECT arxiv_id, title, abstract
        FROM '{path}'
        WHERE arxiv_id IN ({ids_str})
    """

    try:
        result = con.execute(sql).fetchall()
        id_to_abstract = {row[0]: {"title": row[1], "abstract": row[2]} for row in result}
    except Exception:
        id_to_abstract = {}

    # Build result list preserving order
    papers = []
    for entry in paper_entries:
        match = re.match(r"\[([^\]]+)\]\s*(.*)", entry)
        if match:
            arxiv_id = match.group(1)
            title = match.group(2)
            info = id_to_abstract.get(arxiv_id, {})
            papers.append({
                "arxiv_id": arxiv_id,
                "title": info.get("title", title),
                "abstract": info.get("abstract", ""),
                "entry": entry,
            })
        else:
            papers.append({"entry": entry, "abstract": ""})

    return papers


@dataclass
class Cluster:
    """A cluster of papers with a description."""
    name: str
    papers: list[dict]

    def to_xml(self, include_abstracts: bool = False) -> str:
        """Format cluster as XML for prompts."""
        if include_abstracts:
            paper_lines = []
            for p in self.papers:
                abstract = p.get("abstract", "")[:200]
                if abstract:
                    paper_lines.append(f"- [{p['arxiv_id']}] {p['title']}\n  {abstract}...")
                else:
                    paper_lines.append(f"- [{p['arxiv_id']}] {p['title']}")
            papers_str = "\n".join(paper_lines)
        else:
            papers_str = "\n".join(f"- [{p['arxiv_id']}] {p['title']}" for p in self.papers)
        return f'<cluster name="{self.name}">\n{papers_str}\n</cluster>'

    @classmethod
    def from_xml(cls, xml: str, paper_lookup: dict[str, dict]) -> "Cluster":
        """Parse cluster from XML format."""
        name_match = re.search(r'<cluster\s+name="([^"]*)"', xml)
        name = name_match.group(1) if name_match else "Unknown"

        # Extract paper entries
        papers = []
        for line in xml.split("\n"):
            match = re.search(r"-\s*\[([^\]]+)\]", line)
            if match:
                arxiv_id = match.group(1)
                if arxiv_id in paper_lookup:
                    papers.append(paper_lookup[arxiv_id])

        return cls(name=name, papers=papers)


def make_fold_prompt(left_clusters: str, right_clusters: str) -> str:
    """Create prompt for merging clusters in fold step."""
    return f"""You are merging paper clusters. Your output must contain EVERY paper from both inputs - never drop papers.

For each incoming cluster:
1. Score its fit (1-10) against each existing cluster based on thematic overlap
2. If best fit ≥ 7: merge into that cluster. Adjust the heading only if necessary to encompass both.
3. If best fit 4-6: consider whether the incoming cluster should split - some papers may fit an existing cluster while others form a new one.
4. If best fit ≤ 3: create a new cluster.

Output format (use EXACTLY this format):
<cluster name="[Descriptive heading]">
- [arxiv_id] Paper title
- [arxiv_id] Paper title
</cluster>

CRITICAL: Count papers before and after. Input count must equal output count.

<existing_clusters>
{left_clusters}
</existing_clusters>

<incoming_clusters>
{right_clusters}
</incoming_clusters>"""


def make_consolidate_prompt(clusters: str, paper_count: int) -> str:
    """Create prompt for consolidation pass."""
    return f"""Review these clusters for consistency. You may:
- Merge clusters that are too similar (>70% thematic overlap)
- Split clusters that are too broad (papers within don't clearly relate)
- Rename clusters for clarity

Do NOT drop any papers. Every paper in the input must appear exactly once in the output.

Output the same <cluster name="..."> format.

Input paper count: {paper_count}
Your output must contain exactly {paper_count} papers.

<clusters>
{clusters}
</clusters>"""


def parse_clusters_from_response(response: str, paper_lookup: dict[str, dict]) -> list[Cluster]:
    """Parse LLM response into list of Cluster objects."""
    clusters = []
    # Find all cluster blocks
    pattern = r'<cluster\s+name="[^"]*">.*?</cluster>'
    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
        cluster = Cluster.from_xml(match, paper_lookup)
        if cluster.papers:  # Only keep clusters with papers
            clusters.append(cluster)

    return clusters


def clusters_to_xml(clusters: list[Cluster], include_abstracts: bool = False) -> str:
    """Convert list of clusters to XML string."""
    return "\n\n".join(c.to_xml(include_abstracts) for c in clusters)


async def cluster_papers(
    papers: list[dict],
    client: BatchOpenAI,
    show_progress: bool = True,
    chunk_size: int = 10,
) -> list[Cluster]:
    """
    Cluster papers using parallel fold with LLM-based merging.

    Args:
        papers: List of paper dicts with arxiv_id, title, abstract
        client: BatchOpenAI client
        show_progress: Whether to show progress bar
        chunk_size: Papers per initial cluster

    Returns:
        List of Cluster objects
    """
    if not papers:
        return []

    # Build lookup for parsing
    paper_lookup = {p["arxiv_id"]: p for p in papers}

    # Create initial single-paper clusters
    initial_clusters = [
        Cluster(name=p["title"][:50], papers=[p])
        for p in papers
    ]

    # Chunk into groups for initial fold
    chunks = []
    for i in range(0, len(initial_clusters), chunk_size):
        chunk = initial_clusters[i:i + chunk_size]
        chunks.append(chunk)

    pbar = tqdm(
        total=len(chunks) - 1,  # fold needs n-1 combines
        desc="Clustering",
        unit="merge",
        disable=not show_progress,
    )

    async def merge_cluster_groups(left: list[Cluster], right: list[Cluster]) -> list[Cluster]:
        """Merge two groups of clusters."""
        # Count input papers
        left_count = sum(len(c.papers) for c in left)
        right_count = sum(len(c.papers) for c in right)
        expected_count = left_count + right_count

        # Only include abstracts for smaller merges to stay within token limits
        include_abstracts = expected_count <= 50

        left_xml = clusters_to_xml(left, include_abstracts=include_abstracts)
        right_xml = clusters_to_xml(right, include_abstracts=include_abstracts)

        prompt = make_fold_prompt(left_xml, right_xml)

        response = await client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )

        response_text = response.choices[0].message.content or ""
        merged = parse_clusters_from_response(response_text, paper_lookup)

        pbar.update(1)

        # Validate paper count
        merged_count = sum(len(c.papers) for c in merged)

        # Fallback: if parsing failed or lost papers, just concatenate
        if not merged or merged_count < expected_count * 0.9:
            return left + right

        return merged

    # Run parallel fold
    if len(chunks) == 1:
        result = chunks[0]
    else:
        result = await fold(chunks, merge_cluster_groups)

    pbar.close()

    # Consolidation pass
    if show_progress:
        print("Running consolidation pass...")

    consolidated = await consolidate_clusters(
        result, paper_lookup, len(papers), client
    )

    return consolidated


async def consolidate_clusters(
    clusters: list[Cluster],
    paper_lookup: dict[str, dict],
    paper_count: int,
    client: BatchOpenAI,
) -> list[Cluster]:
    """Run consolidation pass to fix drift and granularity issues."""
    # Use titles only (no abstracts) to stay within token limits
    clusters_xml = clusters_to_xml(clusters, include_abstracts=False)
    prompt = make_consolidate_prompt(clusters_xml, paper_count)

    response = await client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16384,
    )

    response_text = response.choices[0].message.content or ""
    consolidated = parse_clusters_from_response(response_text, paper_lookup)

    # Validate paper count
    consolidated_count = sum(len(c.papers) for c in consolidated)
    if consolidated_count < paper_count * 0.9:  # Allow 10% loss tolerance
        # Consolidation lost too many papers, skip it
        return clusters

    # Fallback to original if parsing failed
    if not consolidated:
        return clusters

    return consolidated

"""CLI for arXiv paper sorter."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

from arxiv_sorter.core import (
    search_papers,
    filter_papers,
    rank_papers,
    create_batch_client,
    ComparisonTrace,
    parse_paper_list,
    lookup_abstracts,
    cluster_papers,
    Cluster,
)


logger.remove()
logger.add(sys.stderr, level="INFO", format="<dim>{time:HH:mm:ss}</dim> | {message}")


@click.group()
@click.version_option()
def cli():
    """Rank arXiv papers by relevance using parallel LLM sorting."""
    pass


@cli.command()
@click.argument("query")
@click.option(
    "-i",
    "--interest",
    required=True,
    help="Research interest description or path to file containing it",
)
@click.option(
    "-s",
    "--subject",
    default=None,
    help="Filter by arXiv subject (e.g., 'cs.', 'cs.LG')",
)
@click.option(
    "-n", "--limit", default=100, show_default=True, help="Maximum papers to search"
)
@click.option(
    "-t",
    "--top",
    default=50,
    show_default=True,
    help="Number of top results to display",
)
@click.option(
    "--parquet",
    default=None,
    help="Path to arXiv parquet file (default: from env or workspace root)",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
@click.option(
    "--algorithm",
    type=click.Choice(["quicksort", "mergesort"]),
    default="quicksort",
    show_default=True,
    help="Sorting algorithm to use",
)
@click.option(
    "--trace",
    default=None,
    type=click.Path(),
    help="Output trace file (JSON) for debugging comparisons",
)
@click.option(
    "--filter/--no-filter",
    default=True,
    show_default=True,
    help="Filter out irrelevant papers before ranking",
)
def rank(
    query: str,
    interest: str,
    subject: str | None,
    limit: int,
    top: int,
    parquet: str | None,
    quiet: bool,
    algorithm: str,
    trace: str | None,
    filter: bool,
):
    """Search and rank papers by relevance to your research interest.

    QUERY is the search term to find papers (searches title and abstract).

    Examples:

        arxiv-sorter rank "attention mechanisms" -i "transformer efficiency"

        arxiv-sorter rank "LLM" -i interests.txt -s cs.CL -n 500

        arxiv-sorter rank "inference" -i interests.txt --trace trace.json
    """
    # Load interest from file if it's a path
    interest_text = interest
    try:
        if Path(interest).exists():
            interest_text = Path(interest).read_text().strip()
            if not quiet:
                logger.info(f"Loaded research interest from {interest}")
    except OSError:
        pass  # String too long to be a valid path

    asyncio.run(
        _rank_async(
            query, interest_text, subject, limit, top, parquet, quiet, algorithm, trace, filter
        )
    )


async def _rank_async(
    query: str,
    interest: str,
    subject: str | None,
    limit: int,
    top: int,
    parquet: str | None,
    quiet: bool,
    algorithm: str,
    trace_path: str | None,
    do_filter: bool,
):
    """Async implementation of rank command."""
    if not quiet:
        logger.info(f"Searching for '{query}' papers...")

    papers = search_papers(query, subject=subject, limit=limit, parquet_path=parquet)

    if not quiet:
        logger.info(f"Found {len(papers)} papers")

    if not papers:
        logger.warning("No papers found matching query")
        return

    client = create_batch_client()

    # Filter irrelevant papers first
    if do_filter:
        if not quiet:
            logger.info("Filtering irrelevant papers...")
        papers = await filter_papers(papers, interest, client, show_progress=not quiet)
        if not quiet:
            logger.info(f"{len(papers)} relevant papers remain")

    if not papers:
        logger.warning("No relevant papers found")
        return

    if not quiet:
        logger.info(f"Ranking papers using {algorithm} with pairwise comparisons...")

    # Set up tracing if requested
    traces: list[dict] = []
    trace_counter = [0]

    def on_trace(t: ComparisonTrace) -> None:
        trace_counter[0] += 1
        traces.append({
            "comparison_id": trace_counter[0],
            "timestamp": datetime.now().isoformat(),
            "paper_a_id": t.paper_a["arxiv_id"],
            "paper_b_id": t.paper_b["arxiv_id"],
            "prompt": t.prompt,
            "response": t.response,
            "winner": t.winner,
        })

    ranked = await rank_papers(
        papers,
        interest,
        client,
        algorithm=algorithm,
        show_progress=not quiet,
        on_trace=on_trace if trace_path else None,
    )

    # Write trace file if requested
    if trace_path:
        trace_data = {
            "metadata": {
                "query": query,
                "interest": interest,
                "subject": subject,
                "limit": limit,
                "paper_count": len(papers),
                "algorithm": algorithm,
                "timestamp": datetime.now().isoformat(),
            },
            "papers": papers,
            "comparisons": traces,
            "final_ranking": [p["arxiv_id"] for p in ranked],
        }
        Path(trace_path).write_text(json.dumps(trace_data, indent=2, default=str))
        if not quiet:
            logger.info(f"Wrote {len(traces)} comparisons to {trace_path}")

    if not quiet:
        logger.info("=" * 60)
        logger.info("RANKED PAPERS (most relevant first)")
        logger.info("=" * 60)

    for i, paper in enumerate(ranked[:top], 1):
        click.echo(f"{i:3}. [{paper['arxiv_id']}] {paper['title']}")


@cli.command()
@click.argument("query")
@click.option("-s", "--subject", default=None, help="Filter by arXiv subject")
@click.option(
    "-n", "--limit", default=20, show_default=True, help="Maximum papers to return"
)
@click.option("--parquet", default=None, help="Path to arXiv parquet file")
def search(query: str, subject: str | None, limit: int, parquet: str | None):
    """Search papers without ranking (quick preview).

    Examples:

        arxiv-sorter search "transformer" -s cs.LG -n 10
    """
    papers = search_papers(query, subject=subject, limit=limit, parquet_path=parquet)

    if not papers:
        click.echo("No papers found")
        return

    click.echo(f"Found {len(papers)} papers:\n")
    for paper in papers:
        date = paper.get("submission_date", "unknown")
        click.echo(f"[{paper['arxiv_id']}] ({date}) {paper['title']}")
        click.echo(f"  Subject: {paper['primary_subject']}")
        click.echo()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Output file (default: stdout)",
)
@click.option(
    "--parquet",
    default=None,
    help="Path to arXiv parquet file for abstract lookup",
)
@click.option(
    "-c",
    "--chunk-size",
    default=10,
    show_default=True,
    help="Papers per initial cluster chunk",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format",
)
def cluster(
    input_file: str,
    output: str | None,
    parquet: str | None,
    chunk_size: int,
    quiet: bool,
    output_format: str,
):
    """Cluster papers into thematic groups using LLM-based parallel fold.

    INPUT_FILE should be in output.txt format (numbered list of papers):
        1. [arxiv_id] Paper Title
        2. [arxiv_id] Paper Title
        ...

    Examples:

        arxiv-sorter cluster output.txt -o clusters.txt

        arxiv-sorter cluster papers.txt --format json -o clusters.json

        cat papers.txt | arxiv-sorter cluster /dev/stdin
    """
    asyncio.run(
        _cluster_async(input_file, output, parquet, chunk_size, quiet, output_format)
    )


async def _cluster_async(
    input_file: str,
    output: str | None,
    parquet: str | None,
    chunk_size: int,
    quiet: bool,
    output_format: str,
):
    """Async implementation of cluster command."""
    # Read input file
    input_text = Path(input_file).read_text()
    paper_entries = parse_paper_list(input_text)

    if not paper_entries:
        logger.error("No papers found in input file")
        return

    if not quiet:
        logger.info(f"Found {len(paper_entries)} papers in input")

    # Look up abstracts
    if not quiet:
        logger.info("Looking up paper abstracts...")
    papers = lookup_abstracts(paper_entries, parquet_path=parquet)

    if not quiet:
        with_abstract = sum(1 for p in papers if p.get("abstract"))
        logger.info(f"Found abstracts for {with_abstract}/{len(papers)} papers")

    # Create client and cluster
    client = create_batch_client()

    if not quiet:
        logger.info("Clustering papers...")

    clusters = await cluster_papers(
        papers,
        client,
        show_progress=not quiet,
        chunk_size=chunk_size,
    )

    # Verify paper count
    total_clustered = sum(len(c.papers) for c in clusters)
    if total_clustered != len(papers):
        logger.warning(
            f"Paper count mismatch: input={len(papers)}, clustered={total_clustered}"
        )

    # Format output
    if output_format == "json":
        output_data = {
            "input_count": len(papers),
            "cluster_count": len(clusters),
            "clusters": [
                {
                    "name": c.name,
                    "paper_count": len(c.papers),
                    "papers": [
                        {"arxiv_id": p["arxiv_id"], "title": p["title"]}
                        for p in c.papers
                    ],
                }
                for c in clusters
            ],
        }
        output_text = json.dumps(output_data, indent=2)
    else:
        # Text format
        lines = []
        lines.append(f"# Clustered {len(papers)} papers into {len(clusters)} groups\n")
        for i, c in enumerate(clusters, 1):
            lines.append(f"## {i}. {c.name} ({len(c.papers)} papers)")
            for p in c.papers:
                lines.append(f"  - [{p['arxiv_id']}] {p['title']}")
            lines.append("")
        output_text = "\n".join(lines)

    # Write output
    if output:
        Path(output).write_text(output_text)
        if not quiet:
            logger.info(f"Wrote {len(clusters)} clusters to {output}")
    else:
        click.echo(output_text)


if __name__ == "__main__":
    cli()

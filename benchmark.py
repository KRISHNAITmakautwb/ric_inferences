"""
RIC Benchmark Script — benchmark.py
Measures and compares:
  - Local Ollama (Gemma 3 on Mac Mini M3) inference
  - Cloud OmniRoute inference
  - RIC Pipeline (with routing + caching)

Output:
  - Rich console table
  - PNG charts saved to ./logs/benchmark_results.png
  - JSON report saved to ./logs/benchmark_report.json
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime

import httpx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box
from rich.text import Text

from ric.pipeline import RICPipeline
from ric.inference import InferenceEngine
from ric.utils import load_config, get_logger

logger = get_logger("ric.benchmark")
console = Console()

os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────────
# Benchmark Query Set
# ─────────────────────────────────────────────

BENCHMARK_QUERIES = [
    # --- Simple / Short (should route to LOCAL) ---
    {"id": 1, "query": "What is photosynthesis?",                    "expected": "local",  "category": "simple"},
    {"id": 2, "query": "Define machine learning.",                    "expected": "local",  "category": "simple"},
    {"id": 3, "query": "What is the capital of Japan?",              "expected": "local",  "category": "simple"},
    {"id": 4, "query": "Translate 'Hello' to French.",               "expected": "local",  "category": "simple"},
    {"id": 5, "query": "Summarize briefly: AI is transforming tech.", "expected": "local",  "category": "simple"},

    # --- Complex / Long (should route to CLOUD) ---
    {"id": 6,  "query": "Write a Python script to scrape a website and store results in SQLite with async support.",
                                                                      "expected": "cloud", "category": "complex"},
    {"id": 7,  "query": "Analyze the architectural differences between Transformer and Mamba state space models. Compare their computational complexity and practical use cases.",
                                                                      "expected": "cloud", "category": "complex"},
    {"id": 8,  "query": "Write code for a federated learning system where multiple clients train locally and a server aggregates model weights securely.",
                                                                      "expected": "cloud", "category": "complex"},
    {"id": 9,  "query": "Debug this Python code: def fib(n): return fib(n-1) + fib(n-2). Explain the issue and provide an optimized memoized version with step by step explanation.",
                                                                      "expected": "cloud", "category": "complex"},
    {"id": 10, "query": "Research and explain the architecture of GPT-4, including training methodology, RLHF pipeline, and how it differs from GPT-3 in detail.",
                                                                      "expected": "cloud", "category": "complex"},

    # --- Cache test (same as #1, should be instant) ---
    {"id": 11, "query": "What is photosynthesis?",                   "expected": "cache", "category": "cache"},
    {"id": 12, "query": "Define machine learning.",                   "expected": "cache", "category": "cache"},
]


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class BenchmarkRecord:
    query_id: int
    query: str
    category: str
    expected_dest: str
    actual_dest: str
    routed_correctly: bool
    latency_ms: float
    total_tokens: int
    from_cache: bool
    success: bool
    model: str


# ─────────────────────────────────────────────
# Direct Backend Benchmarks
# ─────────────────────────────────────────────

async def benchmark_local_direct(queries: List[str], engine: InferenceEngine) -> List[Dict]:
    """Benchmark local Ollama directly (bypass router/cache)."""
    results = []
    for q in queries:
        r = await engine.run_local(q)
        results.append({
            "query": q[:40],
            "latency_ms": r.latency_ms,
            "tokens": r.total_tokens,
            "success": r.success,
        })
    return results


async def benchmark_cloud_direct(queries: List[str], engine: InferenceEngine) -> List[Dict]:
    """Benchmark cloud OmniRoute directly (bypass router/cache)."""
    results = []
    for q in queries:
        r = await engine.run_cloud(q)
        results.append({
            "query": q[:40],
            "latency_ms": r.latency_ms,
            "tokens": r.total_tokens,
            "success": r.success,
        })
    return results


# ─────────────────────────────────────────────
# Full Pipeline Benchmark
# ─────────────────────────────────────────────

async def benchmark_pipeline(pipeline: RICPipeline, queries: List[Dict]) -> List[BenchmarkRecord]:
    """Benchmark the full RIC pipeline with routing and caching."""
    records = []
    for q_info in queries:
        result = await pipeline.run(q_info["query"])
        actual_dest = result.destination
        expected = q_info["expected"]
        correct = (
            actual_dest == expected
            or (expected == "cache" and result.from_cache)
            or (expected == "local" and actual_dest == "local")
            or (expected == "cloud" and actual_dest == "cloud")
        )
        records.append(BenchmarkRecord(
            query_id=q_info["id"],
            query=q_info["query"][:50],
            category=q_info["category"],
            expected_dest=expected,
            actual_dest=actual_dest,
            routed_correctly=correct,
            latency_ms=result.latency_ms,
            total_tokens=result.total_tokens,
            from_cache=result.from_cache,
            success=result.success,
            model=result.model,
        ))
    return records


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_results(records: List[BenchmarkRecord], local_direct: List[Dict], cloud_direct: List[Dict]):
    """Generate benchmark charts and save to logs/benchmark_results.png."""
    sns.set_theme(style="darkgrid", palette="deep")
    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")
    fig.suptitle(
        "RIC Inference Pipeline — Benchmark Results\nMac Mini M3 (Gemma 3 12B) vs OmniRoute Cloud",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    colors = {
        "local": "#00d4aa",    # Teal
        "cloud": "#ff6b6b",    # Red-orange
        "cache": "#ffd93d",    # Yellow
    }

    # ── Plot 1: Latency by Destination ───────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#16213e")
    cats = {}
    for r in records:
        dest = r.actual_dest
        cats.setdefault(dest, []).append(r.latency_ms)
    for dest, latencies in cats.items():
        ax1.bar(dest, np.mean(latencies), color=colors.get(dest, "gray"),
                alpha=0.85, edgecolor="white", linewidth=0.5)
        ax1.errorbar(dest, np.mean(latencies), yerr=np.std(latencies),
                     fmt='none', color='white', capsize=5)
    ax1.set_title("Avg Latency by Destination", color="white", fontweight="bold")
    ax1.set_ylabel("Latency (ms)", color="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#16213e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334")

    # ── Plot 2: Latency Distribution (Boxplot) ───────
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#16213e")
    data_for_box = []
    labels_for_box = []
    box_colors = []
    for dest in ["local", "cloud", "cache"]:
        vals = [r.latency_ms for r in records if r.actual_dest == dest]
        if vals:
            data_for_box.append(vals)
            labels_for_box.append(dest)
            box_colors.append(colors[dest])
    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        plt.setp(bp[element], color="white")
    ax2.set_title("Latency Distribution", color="white", fontweight="bold")
    ax2.set_ylabel("Latency (ms)", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334")

    # ── Plot 3: Token Consumption ────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#16213e")
    token_data = {}
    for r in records:
        token_data.setdefault(r.actual_dest, []).append(r.total_tokens)
    dests = list(token_data.keys())
    means = [np.mean(token_data[d]) for d in dests]
    bar_colors = [colors.get(d, "gray") for d in dests]
    bars = ax3.bar(dests, means, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}", ha='center', va='bottom', color='white', fontsize=9)
    ax3.set_title("Avg Token Consumption", color="white", fontweight="bold")
    ax3.set_ylabel("Tokens", color="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#334")

    # ── Plot 4: Routing Accuracy ─────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#16213e")
    correct = sum(1 for r in records if r.routed_correctly)
    incorrect = len(records) - correct
    wedge_colors = ["#00d4aa", "#ff6b6b"]
    wedges, texts, autotexts = ax4.pie(
        [correct, incorrect],
        labels=["Correct", "Incorrect"],
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax4.set_title("Routing Accuracy", color="white", fontweight="bold")

    # ── Plot 5: Cache Hit Rate Over Queries ──────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#16213e")
    hits_cumulative = []
    total_cumulative = []
    for i, r in enumerate(records):
        hits = sum(1 for rec in records[:i+1] if rec.from_cache)
        hits_cumulative.append(hits / (i+1) * 100)
        total_cumulative.append(i+1)
    ax5.plot(total_cumulative, hits_cumulative, color=colors["cache"],
             linewidth=2.5, marker='o', markersize=5)
    ax5.fill_between(total_cumulative, hits_cumulative, alpha=0.2, color=colors["cache"])
    ax5.set_title("Cache Hit Rate Over Queries", color="white", fontweight="bold")
    ax5.set_xlabel("Query #", color="white")
    ax5.set_ylabel("Cache Hit Rate (%)", color="white")
    ax5.tick_params(colors="white")
    ax5.set_ylim(0, 100)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#334")

    # ── Plot 6: Local vs Cloud Direct Latency ────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#16213e")
    q_labels = [f"Q{i+1}" for i in range(min(len(local_direct), len(cloud_direct)))]
    local_lat = [r["latency_ms"] for r in local_direct[:len(q_labels)]]
    cloud_lat = [r["latency_ms"] for r in cloud_direct[:len(q_labels)]]
    x = np.arange(len(q_labels))
    w = 0.35
    ax6.bar(x - w/2, local_lat, w, label="Local (Gemma M3)", color=colors["local"], alpha=0.85)
    ax6.bar(x + w/2, cloud_lat, w, label="Cloud (OmniRoute)", color=colors["cloud"], alpha=0.85)
    ax6.set_title("Direct: Local vs Cloud Latency", color="white", fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(q_labels, color="white", fontsize=7)
    ax6.set_ylabel("Latency (ms)", color="white")
    ax6.tick_params(colors="white")
    ax6.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    for spine in ax6.spines.values():
        spine.set_edgecolor("#334")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = "./logs/benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return out_path


# ─────────────────────────────────────────────
# Rich Console Report
# ─────────────────────────────────────────────

def print_report(records: List[BenchmarkRecord]):
    """Print a rich formatted report table."""
    console.print()
    console.print(Panel(
        "[bold cyan]RIC Inference Pipeline — Benchmark Report[/bold cyan]\n"
        "[dim]Mac Mini M3 · Gemma 3 12B · OmniRoute Cloud[/dim]",
        style="bold blue"
    ))

    table = Table(
        title="Query Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("ID",    width=4,  justify="center")
    table.add_column("Category", width=10)
    table.add_column("Query",    width=40)
    table.add_column("Expected", width=9,  justify="center")
    table.add_column("Actual",   width=9,  justify="center")
    table.add_column("✓",        width=4,  justify="center")
    table.add_column("Cache",    width=7,  justify="center")
    table.add_column("Latency",  width=10, justify="right")
    table.add_column("Tokens",   width=8,  justify="right")

    dest_colors = {"local": "green", "cloud": "red", "cache": "yellow"}

    for r in records:
        ok = "✅" if r.routed_correctly else "❌"
        cache = "HIT" if r.from_cache else "-"
        cache_style = "bold yellow" if r.from_cache else "dim"
        dest_style = dest_colors.get(r.actual_dest, "white")

        table.add_row(
            str(r.query_id),
            r.category,
            r.query[:40] + ("..." if len(r.query) > 40 else ""),
            f"[{dest_colors.get(r.expected_dest, 'white')}]{r.expected_dest}[/]",
            f"[{dest_style}]{r.actual_dest}[/]",
            ok,
            f"[{cache_style}]{cache}[/]",
            f"[cyan]{r.latency_ms:.0f}ms[/]",
            str(r.total_tokens) if r.total_tokens else "[dim]0[/]",
        )

    console.print(table)

    # Summary stats
    correct = sum(1 for r in records if r.routed_correctly)
    cache_hits = sum(1 for r in records if r.from_cache)
    avg_latency = {dest: np.mean([r.latency_ms for r in records if r.actual_dest == dest])
                   for dest in ["local", "cloud", "cache"]}
    total_tokens = sum(r.total_tokens for r in records)
    saved_tokens = sum(r.total_tokens for r in records if r.from_cache)  # approx saved

    summary = Table(title="Summary Statistics", box=box.SIMPLE, show_header=True,
                    header_style="bold cyan", border_style="cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value",  justify="right")

    summary.add_row("Total Queries",          str(len(records)))
    summary.add_row("Routing Accuracy",       f"[green]{correct}/{len(records)} ({correct/len(records)*100:.0f}%)[/]")
    summary.add_row("Cache Hits",             f"[yellow]{cache_hits} ({cache_hits/len(records)*100:.0f}%)[/]")
    summary.add_row("Avg Local Latency",      f"[green]{avg_latency.get('local', 0):.0f} ms[/]")
    summary.add_row("Avg Cloud Latency",      f"[red]{avg_latency.get('cloud', 0):.0f} ms[/]")
    summary.add_row("Avg Cache Latency",      f"[yellow]{avg_latency.get('cache', 0):.0f} ms[/]")
    summary.add_row("Total Tokens Used",      str(total_tokens))
    summary.add_row("Est. Tokens Saved",      f"[cyan]{saved_tokens} (via cache)[/]")

    console.print(summary)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main():
    console.rule("[bold cyan]RIC Benchmark Starting[/bold cyan]")
    console.print(f"[dim]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    pipeline = RICPipeline()
    await pipeline.init()
    engine = InferenceEngine()

    # Simple queries for direct comparison
    simple_queries = [q["query"] for q in BENCHMARK_QUERIES if q["category"] == "simple"]

    records: List[BenchmarkRecord] = []
    local_direct: List[Dict] = []
    cloud_direct: List[Dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # ── Phase 1: Pipeline Benchmark ───────────
        task1 = progress.add_task("[cyan]Running full RIC pipeline...", total=len(BENCHMARK_QUERIES))
        for q_info in BENCHMARK_QUERIES:
            result = await pipeline.run(q_info["query"])
            actual_dest = result.destination
            expected = q_info["expected"]
            correct = (
                actual_dest == expected
                or (expected == "cache" and result.from_cache)
                or (expected == "local" and actual_dest == "local")
                or (expected == "cloud" and actual_dest == "cloud")
            )
            records.append(BenchmarkRecord(
                query_id=q_info["id"],
                query=q_info["query"],
                category=q_info["category"],
                expected_dest=expected,
                actual_dest=actual_dest,
                routed_correctly=correct,
                latency_ms=result.latency_ms,
                total_tokens=result.total_tokens,
                from_cache=result.from_cache,
                success=result.success,
                model=result.model,
            ))
            progress.advance(task1)

        # ── Phase 2: Direct Local Benchmark ──────
        task2 = progress.add_task("[green]Benchmarking local (Gemma/M3)...", total=len(simple_queries))
        for q in simple_queries:
            r = await engine.run_local(q)
            local_direct.append({
                "query": q[:40], "latency_ms": r.latency_ms,
                "tokens": r.total_tokens, "success": r.success,
            })
            progress.advance(task2)

        # ── Phase 3: Direct Cloud Benchmark ──────
        task3 = progress.add_task("[red]Benchmarking cloud (OmniRoute)...", total=len(simple_queries))
        for q in simple_queries:
            r = await engine.run_cloud(q)
            cloud_direct.append({
                "query": q[:40], "latency_ms": r.latency_ms,
                "tokens": r.total_tokens, "success": r.success,
            })
            progress.advance(task3)

    # ── Print Console Report ───────────────────
    print_report(records)

    # ── Generate Charts ────────────────────────
    console.print("\n[cyan]Generating benchmark charts...[/cyan]")
    chart_path = plot_results(records, local_direct, cloud_direct)
    console.print(f"[green]✅ Chart saved: {chart_path}[/green]")

    # ── Save JSON Report ───────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_results": [asdict(r) for r in records],
        "local_direct": local_direct,
        "cloud_direct": cloud_direct,
        "cache_stats": await pipeline.get_stats(),
    }
    report_path = "./logs/benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"[green]✅ JSON report saved: {report_path}[/green]")

    await pipeline.close()
    console.rule("[bold cyan]Benchmark Complete[/bold cyan]")


if __name__ == "__main__":
    asyncio.run(main())

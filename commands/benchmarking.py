"""
Benchmarking and performance testing commands for RuvonInference.

This module provides performance benchmarking capabilities to measure
and demonstrate the effectiveness of optimizations like KV-cache.
"""

import io
import sys
import typer
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from ruvoninference.model.gpt2 import GPT2Model
from ruvoninference.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

from .common import (
    console,
    create_header,
    create_config_panel,
)

app = typer.Typer(help="⚡ Performance benchmarking commands")


@app.command()
def benchmark(
    text: str = typer.Option(
        "Once upon a time", "--text", "-t", help="Starting text prompt"
    ),
    max_length: int = typer.Option(
        20, "--max-length", "-l", help="Number of tokens to generate"
    ),
    runs: int = typer.Option(3, "--runs", "-r", help="Number of benchmark runs"),
    model_name: str = typer.Option(
        "gpt2", "--model", "-m", help="Model to use (gpt2, gpt2-medium, etc.)"
    ),
    device: str = typer.Option(
        "cpu", "--device", "-d", help="Device to run on (cpu or cuda)"
    ),
):
    """
    ⚡ Benchmark KV-cache performance improvement

    This compares generation speed with and without KV-cache optimization,
    demonstrating the dramatic performance improvement from avoiding
    redundant attention computations.
    """

    # Show header
    console.print(create_header())
    console.print()

    # Initialize components with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load tokenizer
        task1 = progress.add_task("🔤 Loading tokenizer...", total=None)
        tokenizer = GPT2TokenizerWrapper(model_name)
        progress.update(task1, completed=True)

        # Load model
        task2 = progress.add_task("🤖 Loading GPT-2 model...", total=None)
        model = GPT2Model(model_name, device=device)
        model.load_model()
        progress.update(task2, completed=True)

    console.print()

    # Show benchmark setup
    config_items = {
        "📝 Prompt": f"'{text}'",
        "🎭 Tokens to generate": str(max_length),
        "🔄 Runs per method": str(runs),
        "🤖 Model": model_name,
    }
    setup_panel = create_config_panel("⚡ KV-Cache Performance Test", config_items)
    console.print(setup_panel)
    console.print()

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors=True)

    # Run benchmark
    console.print("🚀 [bold blue]Running Performance Benchmark...[/bold blue]")
    console.print()

    # Redirect stdout to capture benchmark output
    captured_output = io.StringIO()

    # Temporarily redirect stdout to capture the benchmark prints
    old_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        results = model.benchmark_generation(input_ids, max_length, runs)
    finally:
        sys.stdout = old_stdout

    # Get the captured output and display it
    benchmark_output = captured_output.getvalue()
    console.print(benchmark_output)

    # Create results table
    results_table = Table(
        title="🏆 Performance Comparison", show_header=True, header_style="bold magenta"
    )
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Without KV-Cache", style="red")
    results_table.add_column("With KV-Cache", style="green")
    results_table.add_column("Improvement", style="yellow")

    # Add rows to the table
    results_table.add_row(
        "Total Time",
        f"{results['no_cache_avg_time']:.3f}s",
        f"{results['with_cache_avg_time']:.3f}s",
        f"{results['speedup_factor']:.1f}x faster",
    )

    results_table.add_row(
        "Time per Token",
        f"{results['time_per_token_no_cache']:.3f}s",
        f"{results['time_per_token_with_cache']:.3f}s",
        f"{results['time_per_token_no_cache'] / results['time_per_token_with_cache']:.1f}x faster",
    )

    efficiency = (
        1 - results["with_cache_avg_time"] / results["no_cache_avg_time"]
    ) * 100
    results_table.add_row(
        "Efficiency Gain",
        "100%",
        f"{100 - efficiency:.1f}%",
        f"{efficiency:.1f}% less time",
    )

    console.print(results_table)
    console.print()

    # Performance insights
    if results["speedup_factor"] > 2:
        insight_style = "green"
        insight_text = (
            f"🚀 Excellent! KV-cache provides {results['speedup_factor']:.1f}x speedup"
        )
    elif results["speedup_factor"] > 1.5:
        insight_style = "yellow"
        insight_text = (
            f"⚡ Good! KV-cache provides {results['speedup_factor']:.1f}x speedup"
        )
    else:
        insight_style = "red"
        insight_text = f"🤔 Modest speedup of {results['speedup_factor']:.1f}x - try longer sequences"

    insight_panel = Panel(
        f"{insight_text}\n\n"
        f"💡 [bold white]Why KV-cache works:[/bold white]\n"
        f"• Without cache: Each token requires processing the ENTIRE sequence\n"
        f"• With cache: Only the new token needs processing\n"
        f"• Speedup grows quadratically with sequence length!\n\n"
        f"📈 [bold white]For longer sequences (100+ tokens), expect 10-20x speedup![/bold white]",
        title="🧠 Performance Insights",
        style=insight_style,
        border_style=insight_style,
    )
    console.print(insight_panel)

    # Success message
    console.print()
    success_text = Text()
    success_text.append("✅ ", style="bold green")
    success_text.append("Benchmark Complete!", style="bold white")
    success_text.append(
        f" KV-cache optimization delivers {results['speedup_factor']:.1f}x performance improvement",
        style="dim white",
    )

    console.print(Panel(success_text, style="green", border_style="green"))


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
RuvonVLLM CLI - A beautiful command-line interface for our tiny vLLM inference engine.

This CLI provides an interactive and visually appealing way to demonstrate our
inference engine capabilities, starting with Day 1's tokenizer and forward pass demo.
"""

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ruvonvllm.model.gpt2 import GPT2Model
from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

# Initialize Rich console
console = Console()
app = typer.Typer(help="🚀 RuvonVLLM - Tiny vLLM Inference Engine")


def create_header():
    """Create a beautiful header for the CLI."""
    header_text = Text()
    header_text.append("🚀 ", style="bold red")
    header_text.append("RuvonVLLM", style="bold blue")
    header_text.append(" - Tiny vLLM Inference Engine", style="bold white")

    return Panel(header_text, style="bright_blue", border_style="blue", padding=(1, 2))


def create_model_info_table(model_info: dict) -> Table:
    """Create a beautiful table showing model information."""
    table = Table(
        title="📊 Model Information", show_header=True, header_style="bold magenta"
    )
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow")

    for key, value in model_info.items():
        if key == "parameter_count":
            value = f"{value:,}"
        table.add_row(key.replace("_", " ").title(), str(value))

    return table


def create_tokenization_table(token_ids: list, tokens: list) -> Table:
    """Create a table showing tokenization breakdown."""
    table = Table(
        title="🔤 Tokenization Breakdown", show_header=True, header_style="bold green"
    )
    table.add_column("Position", justify="center", style="cyan", no_wrap=True)
    table.add_column("Token ID", justify="center", style="magenta")
    table.add_column("Token", style="yellow")
    table.add_column("Decoded", style="white")

    tokenizer = GPT2TokenizerWrapper("gpt2")

    for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
        decoded_token = tokenizer.decode([token_id])
        table.add_row(str(i), str(token_id), f"'{token}'", f"'{decoded_token}'")

    return table


def create_predictions_table(
    tokenizer, last_token_logits: torch.Tensor, top_k: int = 5
) -> Table:
    """Create a table showing top predictions."""
    table = Table(title="🎯 Top Predictions", show_header=True, header_style="bold red")
    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Token", style="yellow")
    table.add_column("Token ID", justify="center", style="magenta")
    table.add_column("Logit", justify="right", style="blue")
    table.add_column("Probability", justify="right", style="green")

    top_logits, top_indices = torch.topk(last_token_logits, top_k)

    # Convert raw logits to probabilities using softmax
    # Logits are raw scores (can be any real number, e.g., -67.608, -68.288)
    # Softmax transforms them into interpretable probabilities that sum to 1.0
    # Formula: P(token_i) = e^(logit_i) / Σ(e^(logit_j))
    # Result: -67.608 → 27.5%, -68.288 → 13.9%, etc.
    probabilities = torch.softmax(last_token_logits, dim=0)

    for i, (logit, token_id) in enumerate(zip(top_logits, top_indices)):
        predicted_token = tokenizer.decode([token_id.item()])
        probability = probabilities[
            token_id
        ].item()  # Now we have meaningful percentages

        table.add_row(
            str(i + 1),
            f"'{predicted_token}'",
            str(token_id.item()),
            f"{logit.item():.3f}",
            f"{probability:.1%}",
        )

    return table


@app.command()
def predict(
    text: str = typer.Option("Hello world", "--text", "-t", help="Text to process"),
    model_name: str = typer.Option(
        "gpt2", "--model", "-m", help="Model to use (gpt2, gpt2-medium, etc.)"
    ),
    device: str = typer.Option(
        "cpu", "--device", "-d", help="Device to run on (cpu or cuda)"
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of top predictions to show"
    ),
):
    """
    🎯 Predict next token for given text

    This demonstrates the foundation of our inference engine:
    tokenization → model forward pass → prediction analysis.
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

    # Show input
    input_panel = Panel(
        f"📝 Input Text: [bold yellow]'{text}'[/bold yellow]",
        style="green",
        border_style="green",
    )
    console.print(input_panel)
    console.print()

    # Tokenization
    console.print("🔍 [bold blue]Tokenization Process[/bold blue]")

    # Get detailed tokenization info
    token_ids = tokenizer.encode(text, return_tensors=False)
    tokens = tokenizer.tokenizer.tokenize(text)

    # Show tokenization table
    console.print(create_tokenization_table(token_ids, tokens))
    console.print()

    # Prepare input tensor
    input_ids = tokenizer.encode(text, return_tensors=True)

    # Model forward pass
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("🧠 Running forward pass...", total=None)
        logits = model.forward(input_ids)
        progress.update(task, completed=True)

    console.print()

    # Show logits info
    logits_panel = Panel(
        f"📊 Output Shape: [bold cyan]{logits.shape}[/bold cyan]\n"
        f"📈 Logits Range: [bold cyan][{logits.min().item():.3f}, {logits.max().item():.3f}][/bold cyan]",
        title="🎯 Model Output",
        style="blue",
        border_style="blue",
    )
    console.print(logits_panel)
    console.print()

    # Analyze predictions for the last token
    last_token_logits = logits[0, -1, :]
    console.print(create_predictions_table(tokenizer, last_token_logits, top_k))
    console.print()

    # Show final prediction
    _, top_indices = torch.topk(last_token_logits, 1)
    most_likely_token_id = top_indices[0].item()
    predicted_text = tokenizer.decode([most_likely_token_id])
    complete_text = text + predicted_text

    result_panel = Panel(
        f"📝 Original: [bold yellow]'{text}'[/bold yellow]\n"
        f"🎯 Predicted Next Token: [bold green]'{predicted_text}'[/bold green]\n"
        f"✨ Complete Text: [bold white]'{complete_text}'[/bold white]",
        title="🎉 Prediction Result",
        style="green",
        border_style="green",
    )
    console.print(result_panel)
    console.print()

    # Show model info
    model_info = model.get_model_info()
    console.print(create_model_info_table(model_info))

    # Success message
    console.print()
    success_text = Text()
    success_text.append("✅ ", style="bold green")
    success_text.append("Prediction Complete!", style="bold white")
    success_text.append(
        " Successfully demonstrated: text → tokens → model → logits → predictions",
        style="dim white",
    )

    console.print(Panel(success_text, style="green", border_style="green"))


@app.command()
def generate(
    text: str = typer.Option(
        "Once upon a time", "--text", "-t", help="Starting text prompt"
    ),
    max_length: int = typer.Option(
        20, "--max-length", "-l", help="Number of tokens to generate"
    ),
    model_name: str = typer.Option(
        "gpt2", "--model", "-m", help="Model to use (gpt2, gpt2-medium, etc.)"
    ),
    device: str = typer.Option(
        "cpu", "--device", "-d", help="Device to run on (cpu or cuda)"
    ),
    show_steps: bool = typer.Option(
        False, "--show-steps", "-s", help="Show each generation step"
    ),
):
    """
    🎭 Generate text using greedy decoding

    This demonstrates iterative text generation by repeatedly:
    1. Running forward pass to get next token predictions
    2. Selecting the most likely token (argmax/greedy)
    3. Appending it to the sequence and repeating
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

    # Show input
    input_panel = Panel(
        f"📝 Starting Prompt: [bold yellow]'{text}'[/bold yellow]",
        style="green",
        border_style="green",
    )
    console.print(input_panel)
    console.print()

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors=True)
    console.print(f"🔤 Input tokens: {input_ids.tolist()[0]}")
    console.print()

    # Generate text with progress tracking
    generation_panel = Panel(
        f"🎭 Generating {max_length} tokens using greedy decoding...",
        style="blue",
        border_style="blue",
    )
    console.print(generation_panel)
    console.print()

    if show_steps:
        console.print("🔍 [bold blue]Generation Steps:[/bold blue]")

    # Run generation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"🧠 Generating {max_length} tokens...", total=None)

        # Generate with step-by-step output if requested
        if show_steps:
            generated_tokens = model.generate_greedy(
                input_ids, max_length, show_progress=True
            )
        else:
            generated_tokens = model.generate_greedy(
                input_ids, max_length, show_progress=False
            )

        progress.update(task, completed=True)

    console.print()

    # Show results
    original_text = text
    full_text = tokenizer.decode(generated_tokens)
    generated_part = full_text[len(original_text) :]

    result_panel = Panel(
        f"📝 Original: [bold yellow]'{original_text}'[/bold yellow]\n"
        f"✨ Generated: [bold green]'{generated_part}'[/bold green]\n"
        f"📖 Complete: [bold white]'{full_text}'[/bold white]",
        title="🎉 Generation Result",
        style="green",
        border_style="green",
    )
    console.print(result_panel)
    console.print()

    # Show generation stats
    stats_table = Table(
        title="📊 Generation Statistics", show_header=True, header_style="bold magenta"
    )
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="yellow")

    input_token_count = len(input_ids[0])
    generated_token_count = len(generated_tokens) - input_token_count

    stats_table.add_row("Input Tokens", str(input_token_count))
    stats_table.add_row("Generated Tokens", str(generated_token_count))
    stats_table.add_row("Total Tokens", str(len(generated_tokens)))
    stats_table.add_row("Generation Method", "Greedy (Argmax)")

    console.print(stats_table)
    console.print()

    # Success message
    success_text = Text()
    success_text.append("✅ ", style="bold green")
    success_text.append("Generation Complete!", style="bold white")
    success_text.append(
        f" Successfully generated {generated_token_count} tokens using greedy decoding",
        style="dim white",
    )

    console.print(Panel(success_text, style="green", border_style="green"))


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
    setup_panel = Panel(
        f"🏁 Benchmark Setup:\n"
        f"📝 Prompt: [bold yellow]'{text}'[/bold yellow]\n"
        f"🎭 Tokens to generate: [bold cyan]{max_length}[/bold cyan]\n"
        f"🔄 Runs per method: [bold cyan]{runs}[/bold cyan]\n"
        f"🤖 Model: [bold cyan]{model_name}[/bold cyan]",
        title="⚡ KV-Cache Performance Test",
        style="blue",
        border_style="blue",
    )
    console.print(setup_panel)
    console.print()

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors=True)

    # Run benchmark
    console.print("🚀 [bold blue]Running Performance Benchmark...[/bold blue]")
    console.print()

    # Redirect stdout to capture benchmark output
    import io
    import sys

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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """
    🌐 Start the HTTP API server

    This starts the FastAPI server with streaming text completion endpoints,
    demonstrating Day 4's HTTP server with /completions endpoint.
    """
    import uvicorn

    # Show header
    console.print(create_header())
    console.print()

    # Show server info
    server_panel = Panel(
        f"🌐 Starting RuvonVLLM API Server\n"
        f"📍 Address: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"🔄 Auto-reload: [bold cyan]{'Enabled' if reload else 'Disabled'}[/bold cyan]\n"
        f"📖 API Docs: [bold cyan]http://{host}:{port}/docs[/bold cyan]\n"
        f"🩺 Health Check: [bold cyan]http://{host}:{port}/health[/bold cyan]",
        title="🚀 Day 4: HTTP Server with Streaming",
        style="green",
        border_style="green",
    )
    console.print(server_panel)
    console.print()

    # Instructions
    instructions_text = """
[bold blue]📋 Quick Test Commands:[/bold blue]

[bold yellow]1. Health Check:[/bold yellow]
```bash
curl http://localhost:8000/health
```

[bold yellow]2. Non-streaming completion:[/bold yellow]
```bash
curl -X POST http://localhost:8000/completions \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Once upon a time", "max_tokens": 10}'
```

[bold yellow]3. Streaming completion:[/bold yellow]
```bash
curl -X POST http://localhost:8000/completions \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "The future of AI is", "max_tokens": 15, "stream": true}'
```
    """

    console.print(Panel(instructions_text, style="blue", border_style="blue"))
    console.print()

    # Start the server
    console.print("🚀 [bold green]Starting server...[/bold green]")
    console.print()

    try:
        uvicorn.run(
            "ruvonvllm.api.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n🛑 [bold red]Server stopped by user[/bold red]")
    except Exception as e:
        console.print(f"\n❌ [bold red]Server error: {e}[/bold red]")


@app.command()
def info():
    """
    ℹ️  Show information about RuvonVLLM
    """
    console.print(create_header())
    console.print()

    info_text = """
[bold blue]🚀 RuvonVLLM - Tiny vLLM Inference Engine[/bold blue]

A miniature but real inference system for transformer models, built from scratch
over 20 days. This project demonstrates modern LLM serving techniques including:

[bold green]📦 Current Features:[/bold green]
• 🔤 GPT-2 tokenization and text processing
• 🤖 Model loading and forward pass execution
• 🎯 Logits analysis and next-token prediction
• 🎭 Greedy text generation (iterative decoding)
• ⚡ KV-cache optimization for 10-20x speedup
• 🏁 Performance benchmarking tools
• 💻 Beautiful CLI interface with Rich + Typer
• 🌐 HTTP API server with streaming (/completions)

[bold yellow]🎯 Upcoming Features:[/bold yellow]
• 🎲 Advanced sampling strategies (Day 5)
• ⚡ Continuous batching (Days 6-8)
• 📊 Telemetry and metrics (Day 10)
• 🚀 FlashAttention integration (Day 11)
• 🔧 Custom Triton kernels (Day 12)
• 📄 Paged KV-cache allocator (Day 13)
• 🎭 Speculative decoding (Day 16)
• ⚖️  Quantization support (Day 17)

[bold cyan]Usage:[/bold cyan]
```
# Predict next token with default "Hello world"
python cli.py predict

# Generate text with default "Once upon a time"
python cli.py generate

# Generate with custom prompt
python cli.py generate --text "The future of AI is" --max-length 15

# Show step-by-step generation
python cli.py generate --text "Science is" --show-steps

# Benchmark KV-cache performance
python cli.py benchmark --max-length 30

# Start HTTP API server
python cli.py serve --port 8000

# Use different model
python cli.py generate --model gpt2-medium --text "In a galaxy far, far away"
```
    """

    console.print(Panel(info_text, style="blue", border_style="blue"))


if __name__ == "__main__":
    app()

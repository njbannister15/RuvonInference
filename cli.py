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
def sample(
    text: str = typer.Option(
        "The future of AI is", "--text", "-t", help="Starting text prompt"
    ),
    max_length: int = typer.Option(
        15, "--max-length", "-l", help="Number of tokens to generate"
    ),
    temperature: float = typer.Option(
        0.8, "--temperature", "-temp", help="Temperature (0.1=focused, 2.0=creative)"
    ),
    top_k: int = typer.Option(
        None, "--top-k", "-k", help="Top-k sampling (None=disabled)"
    ),
    top_p: float = typer.Option(
        None, "--top-p", "-p", help="Nucleus sampling (None=disabled)"
    ),
    model_name: str = typer.Option(
        "gpt2", "--model", "-m", help="Model to use (gpt2, gpt2-medium, etc.)"
    ),
    device: str = typer.Option(
        "cpu", "--device", "-d", help="Device to run on (cpu or cuda)"
    ),
    num_samples: int = typer.Option(
        3, "--samples", "-n", help="Number of samples to generate"
    ),
    show_steps: bool = typer.Option(
        False, "--show-steps", "-s", help="Show detailed sampling information"
    ),
):
    """
    🎭 Generate creative text using advanced sampling strategies

    This demonstrates Day 5's sampling techniques: temperature, top-k, and nucleus
    sampling to create more diverse and creative text generation beyond greedy decoding.
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

    # Show sampling configuration
    config_panel = Panel(
        f"📝 Prompt: [bold yellow]'{text}'[/bold yellow]\n"
        f"🎭 Max tokens: [bold cyan]{max_length}[/bold cyan]\n"
        f"🌡️  Temperature: [bold cyan]{temperature}[/bold cyan] ({'focused' if temperature < 0.8 else 'creative' if temperature > 1.2 else 'balanced'})\n"
        f"🔝 Top-k: [bold cyan]{top_k if top_k else 'disabled'}[/bold cyan]\n"
        f"🎯 Top-p: [bold cyan]{top_p if top_p else 'disabled'}[/bold cyan]\n"
        f"📊 Samples: [bold cyan]{num_samples}[/bold cyan]",
        title="🎭 Sampling Configuration",
        style="blue",
        border_style="blue",
    )
    console.print(config_panel)
    console.print()

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors=True)

    # Generate samples
    console.print(
        "🎭 [bold blue]Generating samples with advanced sampling...[/bold blue]"
    )
    console.print()

    for i in range(num_samples):
        console.print(f"[bold green]Sample {i + 1}:[/bold green]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"🧠 Generating sample {i + 1}...", total=None)

            generated_tokens = model.generate_with_sampling(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=True,
                show_progress=show_steps,
            )

            progress.update(task, completed=True)

        # Show result
        full_text = tokenizer.decode(generated_tokens)
        generated_part = full_text[len(text) :]

        result_text = f"[bold yellow]'{text}'[/bold yellow][bold white]{generated_part}[/bold white]"
        console.print(f"  {result_text}")
        console.print()

    # Show sampling comparison if multiple strategies
    if num_samples == 1:
        console.print(
            "💡 [dim]Try increasing --samples to see variety, or adjust temperature/top-k/top-p for different creativity levels[/dim]"
        )
    else:
        console.print(
            "🎯 [bold blue]Notice the variety![/bold blue] Each sample uses the same prompt but different random choices."
        )

    console.print()

    # Success message
    success_text = Text()
    success_text.append("✅ ", style="bold green")
    success_text.append("Sampling Complete!", style="bold white")
    success_text.append(
        f" Generated {num_samples} creative samples using advanced sampling strategies",
        style="dim white",
    )

    console.print(Panel(success_text, style="green", border_style="green"))


@app.command()
def compare(
    text: str = typer.Option(
        "In a world where", "--text", "-t", help="Starting text prompt"
    ),
    max_length: int = typer.Option(
        10, "--max-length", "-l", help="Number of tokens to generate"
    ),
    model_name: str = typer.Option(
        "gpt2", "--model", "-m", help="Model to use (gpt2, gpt2-medium, etc.)"
    ),
    device: str = typer.Option(
        "cpu", "--device", "-d", help="Device to run on (cpu or cuda)"
    ),
):
    """
    🔬 Compare different sampling strategies side-by-side

    This demonstrates the differences between greedy decoding and various
    creative sampling approaches, showing how each strategy affects output diversity.
    """

    # Show header
    console.print(create_header())
    console.print()

    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task1 = progress.add_task("🔤 Loading tokenizer...", total=None)
        tokenizer = GPT2TokenizerWrapper(model_name)
        progress.update(task1, completed=True)

        task2 = progress.add_task("🤖 Loading GPT-2 model...", total=None)
        model = GPT2Model(model_name, device=device)
        model.load_model()
        progress.update(task2, completed=True)

    console.print()

    # Show input
    input_panel = Panel(
        f"📝 Comparing strategies on: [bold yellow]'{text}'[/bold yellow]",
        style="green",
        border_style="green",
    )
    console.print(input_panel)
    console.print()

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors=True)

    # Run comparison
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("🔬 Running strategy comparison...", total=None)
        results = model.compare_sampling_strategies(
            input_ids, max_length, num_samples=3
        )
        progress.update(task, completed=True)

    console.print()

    # Create comparison table
    comparison_table = Table(
        title="🔬 Sampling Strategy Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    comparison_table.add_column("Strategy", style="cyan", no_wrap=True)
    comparison_table.add_column("Sample 1", style="yellow")
    comparison_table.add_column("Sample 2", style="yellow")
    comparison_table.add_column("Sample 3", style="yellow")

    strategy_descriptions = {
        "greedy": "Almost deterministic (temp=0.1)",
        "low_temp": "Focused (temp=0.7)",
        "medium_temp": "Balanced (temp=1.0)",
        "high_temp": "Creative (temp=1.5)",
        "top_k_20": "Conservative variety (k=20)",
        "top_k_50": "Moderate variety (k=50)",
        "top_p_90": "Nucleus 90% (p=0.9)",
        "top_p_95": "Nucleus 95% (p=0.95)",
        "nucleus": "Combined (k=40, p=0.9)",
    }

    for strategy_name, samples in results.items():
        description = strategy_descriptions.get(strategy_name, strategy_name)
        comparison_table.add_row(
            f"{strategy_name}\n[dim]{description}[/dim]",
            f"'{samples[0]}'",
            f"'{samples[1]}'",
            f"'{samples[2]}'",
        )

    console.print(comparison_table)
    console.print()

    # Insights
    insights_text = """
[bold blue]🧠 Key Insights:[/bold blue]

[bold yellow]Temperature:[/bold yellow]
• Low (0.1-0.7): More predictable, coherent text
• Medium (0.8-1.2): Balanced creativity and coherence
• High (1.3-2.0): More surprising, diverse outputs

[bold yellow]Top-k:[/bold yellow]
• Low k (1-20): Conservative, likely tokens only
• High k (50+): Allows more creative word choices

[bold yellow]Top-p (Nucleus):[/bold yellow]
• Dynamically adjusts based on model confidence
• p=0.9: Only most probable tokens that sum to 90%
• p=0.95: Slightly more inclusive vocabulary

[bold green]💡 Pro tip:[/bold green] Combine strategies! Use temperature + top-k/top-p for optimal results.
    """

    console.print(Panel(insights_text, style="blue", border_style="blue"))

    # Success message
    console.print()
    success_text = Text()
    success_text.append("✅ ", style="bold green")
    success_text.append("Comparison Complete!", style="bold white")
    success_text.append(
        " Demonstrated how different sampling strategies create variety in text generation",
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
def stress_test(
    max_requests: int = typer.Option(
        100, "--max-requests", "-n", help="Total number of requests to send"
    ),
    batch_size: int = typer.Option(
        10, "--batch-size", "-b", help="Starting batch size (10, 20, 30, etc.)"
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    prompt: str = typer.Option(
        "Stress test prompt", "--prompt", "-t", help="Prompt to use for requests"
    ),
    max_tokens: int = typer.Option(
        5, "--max-tokens", "-m", help="Number of tokens to generate per request"
    ),
    auto_start_server: bool = typer.Option(
        True, "--auto-start", "-a", help="Automatically start server if not running"
    ),
):
    """
    🚀 Run stress test with increasing batch sizes

    This sends requests in increasing batches (10, 20, 30, 40...) to test
    the queue system's ability to handle multiple concurrent requests.
    """
    import asyncio
    import aiohttp
    import time
    import subprocess
    import signal
    import os
    from rich.live import Live

    # Show header
    console.print(create_header())
    console.print()

    # Show test configuration
    config_panel = Panel(
        f"🚀 Stress Test Configuration:\n"
        f"📊 Total requests: [bold cyan]{max_requests}[/bold cyan]\n"
        f"📦 Batch size: [bold cyan]{batch_size}[/bold cyan] (increments by {batch_size})\n"
        f"🌐 Server: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"📝 Prompt: [bold yellow]'{prompt}'[/bold yellow]\n"
        f"🎭 Tokens per request: [bold cyan]{max_tokens}[/bold cyan]\n"
        f"🔄 Auto-start server: [bold cyan]{'Yes' if auto_start_server else 'No'}[/bold cyan]",
        title="🧪 Stress Test Setup",
        style="blue",
        border_style="blue",
    )
    console.print(config_panel)
    console.print()

    server_process = None

    try:
        # Check if server is running, start if needed
        async def check_server():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/health", timeout=3
                    ) as response:
                        return response.status == 200
            except Exception:
                return False

        async def start_server_if_needed():
            nonlocal server_process
            is_running = await check_server()

            if not is_running and auto_start_server:
                console.print("🚀 [bold blue]Starting API server...[/bold blue]")

                # Start server in background
                server_process = subprocess.Popen(
                    ["make", "run-api"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,  # Create new process group
                )

                # Wait for server to start
                for i in range(30):  # Wait up to 30 seconds
                    await asyncio.sleep(1)
                    if await check_server():
                        console.print(
                            "✅ [bold green]Server started successfully[/bold green]"
                        )
                        console.print()
                        return True

                console.print("❌ [bold red]Failed to start server[/bold red]")
                return False
            elif is_running:
                console.print("✅ [bold green]Server is already running[/bold green]")
                console.print()
                return True
            else:
                console.print(
                    "❌ [bold red]Server not running. Use --auto-start to start automatically[/bold red]"
                )
                return False

        # Send a batch of requests
        async def send_batch(session, batch_size, batch_num):
            """Send a batch of requests concurrently and measure timing."""
            request_data = {
                "prompt": f"{prompt} (batch {batch_num})",
                "max_tokens": max_tokens,
            }

            start_time = time.time()
            tasks = []

            for i in range(batch_size):
                task = session.post(
                    f"http://{host}:{port}/completions",
                    json=request_data,
                    timeout=60,  # 60 second timeout per request
                )
                tasks.append(task)

            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                # Count successful responses
                successful = 0
                failed = 0
                for response in responses:
                    if isinstance(response, Exception):
                        failed += 1
                    else:
                        if response.status == 200:
                            successful += 1
                        else:
                            failed += 1
                        response.close()

                return {
                    "batch_size": batch_size,
                    "batch_num": batch_num,
                    "successful": successful,
                    "failed": failed,
                    "total_time": end_time - start_time,
                    "requests_per_second": batch_size / (end_time - start_time)
                    if end_time > start_time
                    else 0,
                }

            except Exception as e:
                return {
                    "batch_size": batch_size,
                    "batch_num": batch_num,
                    "successful": 0,
                    "failed": batch_size,
                    "total_time": time.time() - start_time,
                    "requests_per_second": 0,
                    "error": str(e),
                }

        async def run_stress_test():
            # Start server if needed
            if not await start_server_if_needed():
                return

            # Initialize session
            async with aiohttp.ClientSession() as session:
                # Get initial queue stats
                try:
                    async with session.get(f"http://{host}:{port}/queue") as response:
                        initial_stats = await response.json()
                        console.print(f"📊 Initial queue state: {initial_stats}")
                        console.print()
                except Exception:
                    console.print("⚠️  Could not fetch initial queue stats")
                    console.print()

                # Run batches
                total_sent = 0
                total_successful = 0
                total_failed = 0
                results = []

                current_batch_size = batch_size
                batch_num = 1

                console.print("🚀 [bold blue]Starting stress test...[/bold blue]")
                console.print()

                # Create results table
                results_table = Table(
                    title="📊 Real-time Stress Test Results",
                    show_header=True,
                    header_style="bold magenta",
                )
                results_table.add_column("Batch #", style="cyan", no_wrap=True)
                results_table.add_column("Size", style="yellow")
                results_table.add_column("Success", style="green")
                results_table.add_column("Failed", style="red")
                results_table.add_column("Time (s)", style="blue")
                results_table.add_column("Req/s", style="magenta")

                with Live(results_table, refresh_per_second=1) as live:
                    while total_sent < max_requests:
                        # Adjust batch size to not exceed max_requests
                        remaining = max_requests - total_sent
                        actual_batch_size = min(current_batch_size, remaining)

                        # Send batch
                        result = await send_batch(session, actual_batch_size, batch_num)
                        results.append(result)

                        # Update stats
                        total_sent += actual_batch_size
                        total_successful += result["successful"]
                        total_failed += result["failed"]

                        # Add to table
                        results_table.add_row(
                            str(batch_num),
                            str(actual_batch_size),
                            str(result["successful"]),
                            str(result["failed"]),
                            f"{result['total_time']:.2f}",
                            f"{result['requests_per_second']:.1f}",
                        )

                        # Update live display
                        live.update(results_table)

                        # Move to next batch
                        current_batch_size += batch_size
                        batch_num += 1

                        # Small delay between batches
                        await asyncio.sleep(1)

                console.print()

                # Get final queue stats
                try:
                    async with session.get(f"http://{host}:{port}/queue") as response:
                        final_stats = await response.json()
                        console.print(f"📊 Final queue state: {final_stats}")
                        console.print()
                except Exception:
                    console.print("⚠️  Could not fetch final queue stats")
                    console.print()

                # Summary statistics
                total_time = sum(r["total_time"] for r in results)
                avg_requests_per_second = (
                    sum(r["requests_per_second"] for r in results) / len(results)
                    if results
                    else 0
                )

                summary_table = Table(
                    title="📈 Stress Test Summary",
                    show_header=True,
                    header_style="bold green",
                )
                summary_table.add_column("Metric", style="cyan", no_wrap=True)
                summary_table.add_column("Value", style="yellow")

                summary_table.add_row("Total Requests Sent", str(total_sent))
                summary_table.add_row("Successful Requests", str(total_successful))
                summary_table.add_row("Failed Requests", str(total_failed))
                summary_table.add_row(
                    "Success Rate",
                    f"{(total_successful/total_sent*100):.1f}%"
                    if total_sent > 0
                    else "0%",
                )
                summary_table.add_row("Total Batches", str(len(results)))
                summary_table.add_row(
                    "Avg Requests/Second", f"{avg_requests_per_second:.1f}"
                )
                summary_table.add_row("Total Test Time", f"{total_time:.2f}s")

                console.print(summary_table)
                console.print()

                # Success message
                if total_successful == total_sent:
                    status_style = "green"
                    status_msg = "✅ All requests successful!"
                elif total_successful > total_sent * 0.8:
                    status_style = "yellow"
                    status_msg = f"⚠️  {total_failed} requests failed"
                else:
                    status_style = "red"
                    status_msg = f"❌ {total_failed} requests failed"

                success_text = Text()
                success_text.append("🧪 ", style="bold blue")
                success_text.append("Stress Test Complete! ", style="bold white")
                success_text.append(status_msg, style=f"bold {status_style}")

                console.print(
                    Panel(success_text, style=status_style, border_style=status_style)
                )

        # Run the async stress test
        asyncio.run(run_stress_test())

    except KeyboardInterrupt:
        console.print("\n🛑 [bold red]Stress test stopped by user[/bold red]")
    except Exception as e:
        console.print(f"\n❌ [bold red]Stress test error: {e}[/bold red]")
    finally:
        # Clean up server process if we started it
        if server_process:
            console.print("🛑 [bold blue]Stopping server...[/bold blue]")
            try:
                # Kill the process group to ensure all child processes are killed
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            console.print("✅ [bold green]Server stopped[/bold green]")


@app.command()
def rapid_test(
    total_requests: int = typer.Option(
        50, "--total", "-n", help="Total number of requests to send rapidly"
    ),
    concurrent_limit: int = typer.Option(
        20, "--concurrent", "-c", help="Maximum concurrent requests"
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    prompt: str = typer.Option(
        "Rapid test", "--prompt", "-t", help="Prompt to use for requests"
    ),
    max_tokens: int = typer.Option(
        10, "--max-tokens", "-m", help="Number of tokens to generate per request"
    ),
    auto_start_server: bool = typer.Option(
        True, "--auto-start", "-a", help="Automatically start server if not running"
    ),
    monitor_queue: bool = typer.Option(
        True, "--monitor", help="Show real-time queue monitoring"
    ),
):
    """
    🚀 Rapid-fire test to fill the queue and test sequential processing

    This sends requests as fast as possible to actually build up the queue
    and demonstrate the sequential processing behavior. Unlike stress-test,
    this doesn't wait between batches.
    """
    import asyncio
    import aiohttp
    import time
    import subprocess
    import signal
    import os
    from rich.live import Live
    from rich.columns import Columns

    # Show header
    console.print(create_header())
    console.print()

    # Show test configuration
    config_panel = Panel(
        f"🚀 Rapid-Fire Test Configuration:\n"
        f"📊 Total requests: [bold cyan]{total_requests}[/bold cyan]\n"
        f"⚡ Concurrent limit: [bold cyan]{concurrent_limit}[/bold cyan]\n"
        f"🌐 Server: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"📝 Prompt: [bold yellow]'{prompt}'[/bold yellow]\n"
        f"🎭 Tokens per request: [bold cyan]{max_tokens}[/bold cyan]\n"
        f"🔄 Auto-start server: [bold cyan]{'Yes' if auto_start_server else 'No'}[/bold cyan]\n"
        f"📊 Monitor queue: [bold cyan]{'Yes' if monitor_queue else 'No'}[/bold cyan]",
        title="⚡ Rapid Queue Test Setup",
        style="blue",
        border_style="blue",
    )
    console.print(config_panel)
    console.print()

    server_process = None
    start_time = time.time()

    try:
        # Check if server is running, start if needed
        async def check_server():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/health", timeout=3
                    ) as response:
                        return response.status == 200
            except Exception:
                return False

        async def start_server_if_needed():
            nonlocal server_process
            is_running = await check_server()

            if not is_running and auto_start_server:
                console.print("🚀 [bold blue]Starting API server...[/bold blue]")

                server_process = subprocess.Popen(
                    ["make", "run-api"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

                # Wait for server to start
                for i in range(30):
                    await asyncio.sleep(1)
                    if await check_server():
                        console.print(
                            "✅ [bold green]Server started successfully[/bold green]"
                        )
                        console.print()
                        return True

                console.print("❌ [bold red]Failed to start server[/bold red]")
                return False
            elif is_running:
                console.print("✅ [bold green]Server is already running[/bold green]")
                console.print()
                return True
            else:
                console.print(
                    "❌ [bold red]Server not running. Use --auto-start to start automatically[/bold red]"
                )
                return False

        async def get_queue_stats(session):
            """Fetch current queue statistics."""
            try:
                async with session.get(
                    f"http://{host}:{port}/queue", timeout=2
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception:
                return None

        async def send_single_request(session, request_id):
            """Send a single request and track its timing."""
            request_data = {
                "prompt": f"{prompt} #{request_id}",
                "max_tokens": max_tokens,
            }

            start_time_req = time.time()
            try:
                async with session.post(
                    f"http://{host}:{port}/completions", json=request_data, timeout=60
                ) as response:
                    end_time_req = time.time()

                    if response.status == 200:
                        result = await response.json()
                        response_text = ""
                        if "choices" in result and len(result["choices"]) > 0:
                            response_text = result["choices"][0].get("text", "")

                        return {
                            "id": request_id,
                            "status": "completed",
                            "duration": end_time_req - start_time_req,
                            "response": response_text,
                            "completed_at": end_time_req,
                        }
                    else:
                        return {
                            "id": request_id,
                            "status": "failed",
                            "duration": end_time_req - start_time_req,
                            "error": f"HTTP {response.status}",
                            "completed_at": end_time_req,
                        }

            except Exception as e:
                return {
                    "id": request_id,
                    "status": "failed",
                    "duration": time.time() - start_time_req,
                    "error": str(e),
                    "completed_at": time.time(),
                }

        def create_monitoring_display(
            active_requests, completed_requests, current_stats
        ):
            """Create real-time monitoring display."""

            # Stats table
            stats_table = Table(show_header=False, box=None, padding=(0, 1))
            stats_table.add_column("Metric", style="cyan", no_wrap=True)
            stats_table.add_column("Value", style="yellow")

            elapsed = time.time() - start_time
            stats_table.add_row("🕒 Elapsed", f"{elapsed:.1f}s")
            stats_table.add_row("🚀 Active Requests", str(len(active_requests)))
            stats_table.add_row("✅ Completed", str(len(completed_requests)))
            stats_table.add_row(
                "❌ Failed",
                str(sum(1 for r in completed_requests if r["status"] == "failed")),
            )

            if current_stats:
                stats_table.add_row(
                    "📦 Queue Size", str(current_stats.get("queue_size", "N/A"))
                )
                stats_table.add_row(
                    "⏳ Queued", str(current_stats.get("queued_requests", "N/A"))
                )
                stats_table.add_row(
                    "🔄 Processing",
                    str(current_stats.get("processing_requests", "N/A")),
                )
                stats_table.add_row(
                    "📊 Total Processed",
                    str(current_stats.get("total_processed", "N/A")),
                )

            stats_panel = Panel(
                stats_table, title="📊 Real-time Statistics", style="green"
            )

            # Recent completions
            recent_table = Table(show_header=True, header_style="bold magenta")
            recent_table.add_column("ID", style="cyan", no_wrap=True)
            recent_table.add_column("Status", style="yellow")
            recent_table.add_column("Duration", style="blue")
            recent_table.add_column("Response", style="white")

            for req in completed_requests[-8:]:  # Show last 8
                response_text = req.get("response", req.get("error", ""))
                if len(response_text) > 30:
                    response_text = response_text[:27] + "..."

                status_style = "green" if req["status"] == "completed" else "red"
                status_text = f"[{status_style}]{req['status']}[/{status_style}]"

                recent_table.add_row(
                    f"#{req['id']}",
                    status_text,
                    f"{req['duration']:.2f}s",
                    f"'{response_text}'",
                )

            recent_panel = Panel(
                recent_table, title="📝 Recent Completions", style="blue"
            )

            return Columns([stats_panel, recent_panel])

        async def run_rapid_test():
            # Start server if needed
            if not await start_server_if_needed():
                return

            async with aiohttp.ClientSession() as session:
                active_requests = {}
                completed_requests = []
                semaphore = asyncio.Semaphore(concurrent_limit)

                async def limited_request(request_id):
                    async with semaphore:
                        return await send_single_request(session, request_id)

                console.print(
                    "⚡ [bold blue]Sending requests as fast as possible...[/bold blue]"
                )
                console.print()

                if monitor_queue:

                    def create_display():
                        return create_monitoring_display(
                            active_requests, completed_requests, None
                        )

                    with Live(create_display(), refresh_per_second=2) as live:
                        # Create all tasks immediately
                        tasks = []
                        for i in range(total_requests):
                            task = asyncio.create_task(limited_request(i + 1))
                            tasks.append(task)
                            active_requests[i + 1] = task

                        # Monitor completion
                        while active_requests:
                            # Get current queue stats
                            current_stats = await get_queue_stats(session)

                            # Check for completed tasks
                            done_tasks = []
                            for req_id, task in list(active_requests.items()):
                                if task.done():
                                    try:
                                        result = await task
                                        completed_requests.append(result)
                                        done_tasks.append(req_id)
                                    except Exception as e:
                                        completed_requests.append(
                                            {
                                                "id": req_id,
                                                "status": "failed",
                                                "duration": 0,
                                                "error": str(e),
                                                "completed_at": time.time(),
                                            }
                                        )
                                        done_tasks.append(req_id)

                            # Remove completed tasks
                            for req_id in done_tasks:
                                active_requests.pop(req_id, None)

                            # Update display
                            live.update(
                                create_monitoring_display(
                                    active_requests, completed_requests, current_stats
                                )
                            )

                            await asyncio.sleep(0.1)

                        # Wait for all tasks to complete
                        await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Simple mode without monitoring
                    tasks = [limited_request(i + 1) for i in range(total_requests)]
                    completed_requests = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )

                console.print()

                # Final statistics
                total_time = time.time() - start_time
                successful = sum(
                    1
                    for r in completed_requests
                    if isinstance(r, dict) and r.get("status") == "completed"
                )
                failed = len(completed_requests) - successful

                final_table = Table(
                    title="🎯 Final Results",
                    show_header=True,
                    header_style="bold green",
                )
                final_table.add_column("Metric", style="cyan", no_wrap=True)
                final_table.add_column("Value", style="yellow")

                final_table.add_row("Total Requests", str(total_requests))
                final_table.add_row("Successful", str(successful))
                final_table.add_row("Failed", str(failed))
                final_table.add_row(
                    "Success Rate", f"{(successful/total_requests*100):.1f}%"
                )
                final_table.add_row("Total Time", f"{total_time:.2f}s")
                final_table.add_row(
                    "Requests/Second", f"{total_requests/total_time:.1f}"
                )

                console.print(final_table)
                console.print()

                # Get final queue stats
                try:
                    final_stats = await get_queue_stats(session)
                    if final_stats:
                        console.print(f"📊 Final queue state: {final_stats}")
                except Exception:
                    pass

        # Run the rapid test
        asyncio.run(run_rapid_test())

    except KeyboardInterrupt:
        console.print("\n🛑 [bold red]Rapid test stopped by user[/bold red]")
    except Exception as e:
        console.print(f"\n❌ [bold red]Rapid test error: {e}[/bold red]")
    finally:
        # Clean up server process if we started it
        if server_process:
            console.print("🛑 [bold blue]Stopping server...[/bold blue]")
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            console.print("✅ [bold green]Server stopped[/bold green]")


@app.command()
def monitor(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    refresh_rate: float = typer.Option(
        0.5, "--refresh", "-r", help="Refresh rate in seconds"
    ),
    auto_start_server: bool = typer.Option(
        False, "--auto-start", "-a", help="Automatically start server if not running"
    ),
    send_test_requests: bool = typer.Option(
        False, "--test-requests", "-t", help="Send periodic test requests"
    ),
):
    """
    📊 Real-time monitoring dashboard for queue and requests

    This creates a live-updating CLI dashboard showing:
    - Real-time queue statistics
    - Active request tracking
    - Live streaming responses
    - Server health metrics
    """
    import asyncio
    import aiohttp
    import time
    import subprocess
    import signal
    import os
    from datetime import datetime
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich.text import Text

    # Show header
    console.print(create_header())
    console.print()

    # Show monitor configuration
    config_panel = Panel(
        f"📊 Real-time Monitor Configuration:\n"
        f"🌐 Server: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"🔄 Refresh rate: [bold cyan]{refresh_rate}s[/bold cyan]\n"
        f"🚀 Auto-start server: [bold cyan]{'Yes' if auto_start_server else 'No'}[/bold cyan]\n"
        f"🧪 Send test requests: [bold cyan]{'Yes' if send_test_requests else 'No'}[/bold cyan]",
        title="📊 Live Dashboard Setup",
        style="blue",
        border_style="blue",
    )
    console.print(config_panel)
    console.print()

    server_process = None
    active_requests = {}
    request_history = []
    start_time = time.time()

    def create_queue_stats_panel(stats):
        """Create a panel showing current queue statistics."""
        if not stats:
            return Panel("❌ Could not fetch queue stats", style="red")

        # Calculate uptime
        uptime = time.time() - start_time
        uptime_str = (
            f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
        )

        queue_table = Table(show_header=False, box=None, padding=(0, 1))
        queue_table.add_column("Metric", style="cyan", no_wrap=True)
        queue_table.add_column("Value", style="yellow")

        queue_table.add_row("🕒 Uptime", uptime_str)
        queue_table.add_row("📦 Queue Size", str(stats.get("queue_size", "N/A")))
        queue_table.add_row(
            "⏳ Queued Requests", str(stats.get("queued_requests", "N/A"))
        )
        queue_table.add_row(
            "🔄 Processing", str(stats.get("processing_requests", "N/A"))
        )
        queue_table.add_row(
            "✅ Total Processed", str(stats.get("total_processed", "N/A"))
        )
        queue_table.add_row("❌ Total Failed", str(stats.get("total_failed", "N/A")))

        avg_wait = stats.get("average_wait_time")
        avg_proc = stats.get("average_processing_time")
        queue_table.add_row(
            "⏱️  Avg Wait Time", f"{avg_wait:.3f}s" if avg_wait else "N/A"
        )
        queue_table.add_row(
            "⚡ Avg Process Time", f"{avg_proc:.3f}s" if avg_proc else "N/A"
        )

        current_req = stats.get("current_request_id")
        queue_table.add_row(
            "🎯 Current Request", current_req[:8] + "..." if current_req else "None"
        )

        return Panel(queue_table, title="📊 Queue Statistics", style="green")

    def create_active_requests_panel(queue_stats):
        """Create a panel showing server-side queue information and client requests."""
        # Create a table showing server queue status and our own tracked requests
        req_table = Table(show_header=True, header_style="bold magenta")
        req_table.add_column("Source", style="cyan", no_wrap=True)
        req_table.add_column("Status", style="yellow")
        req_table.add_column("Count/Info", style="blue")
        req_table.add_column("Details", style="white")

        if queue_stats:
            # Show server-side queue information
            queue_size = queue_stats.get("queue_size", 0)
            processing = queue_stats.get("processing_requests", 0)
            current_req = queue_stats.get("current_request_id")

            if queue_size > 0:
                req_table.add_row(
                    "🌐 Server Queue",
                    "⏳ queued",
                    str(queue_size),
                    "Requests waiting to be processed",
                )

            if processing > 0:
                req_table.add_row(
                    "🌐 Server",
                    "🔄 processing",
                    str(processing),
                    f"Current: {current_req[:8] + '...' if current_req else 'unknown'}",
                )

            if queue_size == 0 and processing == 0:
                req_table.add_row(
                    "🌐 Server", "💤 idle", "0", "No requests in queue or processing"
                )

        # Show our own tracked requests (if any)
        if active_requests:
            for req_id, req_data in list(active_requests.items())[:5]:  # Show last 5
                elapsed = time.time() - req_data.get("start_time", time.time())
                response_text = req_data.get("response", "")
                if len(response_text) > 30:
                    response_text = response_text[:27] + "..."

                req_table.add_row(
                    "📱 Monitor",
                    req_data.get("status", "unknown"),
                    f"{elapsed:.1f}s",
                    f"'{response_text}'",
                )

        if not queue_stats and not active_requests:
            return Panel("No queue data or active requests", style="dim")

        return Panel(req_table, title="🔄 Request Activity", style="blue")

    def create_request_history_panel(server_completions=None):
        """Create a panel showing recent completed requests from server and local history."""
        history_table = Table(show_header=True, header_style="bold magenta")
        history_table.add_column("Time", style="cyan", no_wrap=True)
        history_table.add_column("Prompt", style="green", no_wrap=True)
        history_table.add_column("Status", style="yellow")
        history_table.add_column("Response", style="white")

        # Combine server completions and local history
        all_requests = []

        # Add server-side completions (external requests)
        if server_completions:
            for req in server_completions:
                if req.get("completed_at"):
                    all_requests.append(
                        {
                            "completed_at": req["completed_at"],
                            "prompt": req.get("prompt", "Unknown"),
                            "response": req.get("response", "No response"),
                            "status": req.get("status", "unknown"),
                            "total_time": req.get("total_time", 0),
                            "source": "server",
                        }
                    )

        # Add local history (monitor's own requests)
        for req in request_history:
            all_requests.append(
                {
                    "completed_at": req["completed_at"],
                    "prompt": "Monitor test",
                    "response": req.get("response", "No response"),
                    "status": req["status"],
                    "total_time": req.get("duration", 0),
                    "source": "monitor",
                }
            )

        if not all_requests:
            return Panel("No completed requests yet", style="dim")

        # Sort by completion time (newest first) and take last 8
        all_requests.sort(key=lambda x: x["completed_at"], reverse=True)

        for req in all_requests[:8]:
            completion_time = datetime.fromtimestamp(req["completed_at"]).strftime(
                "%H:%M:%S"
            )

            # Truncate prompt and response for display
            prompt_text = req["prompt"]
            if len(prompt_text) > 20:
                prompt_text = prompt_text[:17] + "..."

            response_text = req["response"]
            if len(response_text) > 30:
                response_text = response_text[:27] + "..."

            status_style = "green" if req["status"] == "completed" else "red"
            status_text = f"[{status_style}]{req['status']}[/{status_style}]"

            # Add source indicator
            if req.get("source") == "monitor":
                prompt_text = f"📱 {prompt_text}"
            else:
                prompt_text = f"🌐 {prompt_text}"

            history_table.add_row(
                completion_time, prompt_text, status_text, f"'{response_text}'"
            )

        return Panel(
            history_table, title="📝 Recent Requests & Responses", style="yellow"
        )

    def create_server_health_panel(health_data):
        """Create a panel showing server health."""
        if not health_data:
            return Panel("❌ Server offline", style="red")

        health_table = Table(show_header=False, box=None, padding=(0, 1))
        health_table.add_column("Metric", style="cyan", no_wrap=True)
        health_table.add_column("Value", style="yellow")

        health_table.add_row("🩺 Status", health_data.get("status", "unknown"))
        models = health_data.get("models_loaded", [])
        tokenizers = health_data.get("tokenizers_loaded", [])
        health_table.add_row("🤖 Models", f"{len(models)} loaded")
        health_table.add_row("🔤 Tokenizers", f"{len(tokenizers)} loaded")

        return Panel(health_table, title="🩺 Server Health", style="green")

    try:
        # Check if server is running, start if needed
        async def check_server():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/health", timeout=3
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        return None
            except Exception:
                return None

        async def start_server_if_needed():
            nonlocal server_process
            health_data = await check_server()

            if not health_data and auto_start_server:
                console.print("🚀 [bold blue]Starting API server...[/bold blue]")

                server_process = subprocess.Popen(
                    ["make", "run-api"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

                # Wait for server to start
                for i in range(30):
                    await asyncio.sleep(1)
                    health_data = await check_server()
                    if health_data:
                        console.print(
                            "✅ [bold green]Server started successfully[/bold green]"
                        )
                        console.print()
                        return health_data

                console.print("❌ [bold red]Failed to start server[/bold red]")
                return None
            elif health_data:
                console.print("✅ [bold green]Server is running[/bold green]")
                console.print()
                return health_data
            else:
                console.print(
                    "❌ [bold red]Server not running. Use --auto-start to start automatically[/bold red]"
                )
                return None

        async def send_test_request(session, request_num):
            """Send a test request and track its progress."""
            if not send_test_requests:
                return

            try:
                request_data = {
                    "prompt": f"Monitor test {request_num}: Once upon a time",
                    "max_tokens": 8,
                }

                start_time_req = time.time()
                async with session.post(
                    f"http://{host}:{port}/completions", json=request_data, timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        duration = time.time() - start_time_req

                        # Extract response text
                        response_text = ""
                        if "choices" in result and len(result["choices"]) > 0:
                            response_text = result["choices"][0].get("text", "")

                        # Add to history
                        request_history.append(
                            {
                                "completed_at": time.time(),
                                "status": "completed",
                                "duration": duration,
                                "response": response_text,
                            }
                        )

                        # Keep history limited
                        if len(request_history) > 20:
                            request_history.pop(0)

            except Exception as e:
                # Add failed request to history
                request_history.append(
                    {
                        "completed_at": time.time(),
                        "status": "failed",
                        "duration": time.time() - start_time_req,
                        "response": str(e)[:50],
                    }
                )

        async def get_queue_stats(session):
            """Fetch current queue statistics."""
            try:
                async with session.get(
                    f"http://{host}:{port}/queue", timeout=3
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception:
                return None

        async def get_server_completions(session):
            """Fetch recent completed requests from the server."""
            try:
                async with session.get(
                    f"http://{host}:{port}/queue/recent?limit=10", timeout=3
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception:
                return None

        async def run_monitor():
            # Start server if needed
            health_data = await start_server_if_needed()
            if not health_data:
                return

            async with aiohttp.ClientSession() as session:
                request_counter = 1

                def create_layout():
                    # Create layout
                    layout = Layout()

                    layout.split_column(
                        Layout(name="header", size=3),
                        Layout(name="main"),
                        Layout(name="footer", size=3),
                    )

                    layout["main"].split_row(Layout(name="left"), Layout(name="right"))

                    layout["left"].split_column(
                        Layout(name="queue"), Layout(name="health")
                    )

                    layout["right"].split_column(
                        Layout(name="active"), Layout(name="history")
                    )

                    return layout

                def update_layout(layout, stats, health, server_completions=None):
                    # Header
                    header_text = Text()
                    header_text.append("📊 ", style="bold blue")
                    header_text.append("RuvonVLLM Live Monitor", style="bold white")
                    header_text.append(
                        f" • {datetime.now().strftime('%H:%M:%S')}", style="dim white"
                    )
                    layout["header"].update(Align.center(header_text))

                    # Content panels
                    layout["queue"].update(create_queue_stats_panel(stats))
                    layout["health"].update(create_server_health_panel(health))
                    layout["active"].update(create_active_requests_panel(stats))
                    layout["history"].update(
                        create_request_history_panel(server_completions)
                    )

                    # Footer
                    footer_text = Text()
                    footer_text.append("Press ", style="dim white")
                    footer_text.append("Ctrl+C", style="bold red")
                    footer_text.append(" to exit", style="dim white")
                    layout["footer"].update(Align.center(footer_text))

                layout = create_layout()

                with Live(
                    layout, refresh_per_second=1 / refresh_rate, screen=True
                ) as live:
                    try:
                        while True:
                            # Fetch current data
                            stats = await get_queue_stats(session)
                            health = await check_server()
                            server_completions = await get_server_completions(session)

                            # Send test request occasionally
                            if send_test_requests and request_counter % 10 == 0:
                                asyncio.create_task(
                                    send_test_request(session, request_counter // 10)
                                )

                            # Update layout
                            update_layout(layout, stats, health, server_completions)
                            live.update(layout)

                            request_counter += 1
                            await asyncio.sleep(refresh_rate)

                    except KeyboardInterrupt:
                        pass

        # Run the async monitor
        asyncio.run(run_monitor())

    except KeyboardInterrupt:
        console.print("\n🛑 [bold red]Monitor stopped by user[/bold red]")
    except Exception as e:
        console.print(f"\n❌ [bold red]Monitor error: {e}[/bold red]")
    finally:
        # Clean up server process if we started it
        if server_process:
            console.print("🛑 [bold blue]Stopping server...[/bold blue]")
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            console.print("✅ [bold green]Server stopped[/bold green]")


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
• 🎲 Advanced sampling: temperature, top-k, nucleus (top-p)
• 📦 Request queue system for sequential processing
• 🚀 Stress testing with incremental batch sizes
• 📊 Real-time monitoring dashboard (htop for LLM servers)

[bold yellow]🎯 Upcoming Features:[/bold yellow]
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

# Generate creative text with sampling
python cli.py sample --text "The future is" --temperature 1.2 --top-k 40

# Compare different sampling strategies
python cli.py compare --text "In a world where"

# Benchmark KV-cache performance
python cli.py benchmark --max-length 30

# Start HTTP API server
python cli.py serve --port 8000

# Run stress test with 100 requests in batches of 10, 20, 30, 40
python cli.py stress-test --max-requests 100 --batch-size 10

# Real-time monitoring dashboard (like htop for LLM servers)
python cli.py monitor --test-requests --refresh 0.5

# Use different model
python cli.py generate --model gpt2-medium --text "In a galaxy far, far away"
```
    """

    console.print(Panel(info_text, style="blue", border_style="blue"))


if __name__ == "__main__":
    app()

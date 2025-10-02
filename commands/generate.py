"""
Text generation commands for RuvonVLLM.

This module provides various text generation capabilities including:
- Next token prediction
- Greedy text generation
- Advanced sampling strategies (temperature, top-k, nucleus)
- Comparison of different sampling approaches
"""

import torch
import typer
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from ruvonvllm.model.gpt2 import GPT2Model
from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

from .common import (
    console,
    create_header,
)

app = typer.Typer(help="🎭 Text generation and sampling commands")


def create_model_info_table(model_info: dict) -> Table:
    """
    Create a beautiful table showing model information.

    Args:
        model_info: Dictionary containing model information

    Returns:
        Table: Rich table displaying model information
    """
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
    """
    Create a table showing tokenization breakdown.

    Args:
        token_ids: List of token IDs
        tokens: List of token strings

    Returns:
        Table: Rich table showing tokenization details
    """
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
    """
    Create a table showing top predictions.

    Args:
        tokenizer: The tokenizer to use for decoding
        last_token_logits: Logits tensor for the last token
        top_k: Number of top predictions to show

    Returns:
        Table: Rich table showing top predictions with probabilities
    """
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


@app.command(name="gen", help="🎭 Generate text using greedy decoding")
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

    This demonstrates Part 5's sampling techniques: temperature, top-k, and nucleus
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


if __name__ == "__main__":
    app()

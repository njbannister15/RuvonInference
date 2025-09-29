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
• 💻 Beautiful CLI interface with Rich + Typer

[bold yellow]🎯 Upcoming Features:[/bold yellow]
• 🔄 Greedy decode loops (Day 2)
• 💾 KV-cache optimization (Day 3)
• 🌐 HTTP API server (Day 4)
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

# Try different text
python cli.py predict --text "The future of AI is"

# Use different model
python cli.py predict --model gpt2-medium --text "Science is"
```
    """

    console.print(Panel(info_text, style="blue", border_style="blue"))


if __name__ == "__main__":
    app()

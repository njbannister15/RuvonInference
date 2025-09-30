#!/usr/bin/env python3
"""
RuvonVLLM CLI - A beautiful command-line interface for our tiny vLLM inference engine.

This CLI provides an interactive and visually appealing way to demonstrate our
inference engine capabilities, starting with Day 1's tokenizer and forward pass demo.
"""

import typer
import uvicorn
from rich.panel import Panel

from commands.common import console, create_header
from commands import generate, benchmarking, monitoring, testing

# Initialize main CLI app
app = typer.Typer(help="🚀 RuvonVLLM - Tiny vLLM Inference Engine")

# Add command modules
app.add_typer(generate.app, name="generate", help="🎭 Text generation commands")
app.add_typer(
    benchmarking.app, name="benchmark", help="⚡ Performance benchmarking commands"
)
app.add_typer(monitoring.app, name="monitor", help="📊 Real-time monitoring commands")
app.add_typer(testing.app, name="test", help="🧪 Testing and load testing commands")


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
    """🎭 Generate creative text using advanced sampling strategies"""
    generate.sample(
        text,
        max_length,
        temperature,
        top_k,
        top_p,
        model_name,
        device,
        num_samples,
        show_steps,
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    queue_mode: str = typer.Option(
        "batched",
        "--queue-mode",
        "-q",
        help="Queue processing mode: sequential (Part 6), batched (Part 7), continuous (Part 8)",
    ),
    # Keep the old parameter for backwards compatibility
    use_batched_queue: bool = typer.Option(
        None,
        "--use-batched-queue/--no-batched-queue",
        help="[DEPRECATED] Use --queue-mode instead",
        hidden=True,
    ),
):
    """
    🌐 Start the HTTP API server

    This starts the FastAPI server with streaming text completion endpoints,
    demonstrating Day 4's HTTP server with /completions endpoint.
    """
    # Show header
    console.print(create_header())
    console.print()

    # Handle backwards compatibility
    if use_batched_queue is not None:
        queue_mode = "batched" if use_batched_queue else "sequential"
        console.print(
            "[yellow]⚠️  WARNING: --use-batched-queue is deprecated, use --queue-mode instead[/yellow]"
        )

    # Validate queue mode
    valid_modes = ["sequential", "batched", "continuous"]
    if queue_mode not in valid_modes:
        console.print(
            f"[red]❌ Invalid queue mode: {queue_mode}. Valid modes: {', '.join(valid_modes)}[/red]"
        )
        raise typer.Exit(1)

    # Determine queue mode display
    mode_info = {
        "sequential": ("Part 6: Sequential Processing", "yellow"),
        "batched": ("Part 7: Prefill Batching", "cyan"),
        "continuous": ("Part 8: True Continuous Batching", "magenta"),
    }
    queue_display, queue_style = mode_info[queue_mode]

    # Show server info
    server_panel = Panel(
        f"🌐 Starting RuvonVLLM API Server\n"
        f"📍 Address: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"🔄 Auto-reload: [bold cyan]{'Enabled' if reload else 'Disabled'}[/bold cyan]\n"
        f"📦 Queue Mode: [bold {queue_style}]{queue_display}[/bold {queue_style}]\n"
        f"📖 API Docs: [bold cyan]http://{host}:{port}/docs[/bold cyan]\n"
        f"🩺 Health Check: [bold cyan]http://{host}:{port}/health[/bold cyan]",
        title="🚀 RuvonVLLM API Server",
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

    # Set environment variable for queue mode
    import os

    os.environ["QUEUE_MODE"] = queue_mode

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
        console.print_exception()
    except Exception as e:
        console.print(f"\n❌ [bold red]Server error: {e}[/bold red]")
        console.print_exception()


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
# Generate text with default "Once upon a time"
python cli.py generate generate

# Generate with custom prompt
python cli.py generate generate --text "The future of AI is" --max-length 15

# Show step-by-step generation
python cli.py generate generate --text "Science is" --show-steps

# Generate creative text with sampling
python cli.py sample --text "The future is" --temperature 1.2 --top-k 40

# Compare different sampling strategies
python cli.py generate compare --text "In a world where"

# Benchmark KV-cache performance
python cli.py benchmark benchmark --max-length 30

# Start HTTP API server
python cli.py serve --port 8000

# Run stress test with 100 requests in batches of 10, 20, 30, 40
python cli.py test stress-test --max-requests 100 --batch-size 10

# Real-time monitoring dashboard (like htop for LLM servers)
python cli.py monitor monitor --test-requests --refresh 0.5

# Use different model
python cli.py generate generate --model gpt2-medium --text "In a galaxy far, far away"
```
    """

    console.print(Panel(info_text, style="blue", border_style="blue"))


if __name__ == "__main__":
    app()

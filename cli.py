#!/usr/bin/env python3
"""
RuvonInference CLI - A beautiful command-line interface for our educational inference engine.

This CLI provides an interactive and visually appealing way to demonstrate our
inference engine capabilities, starting with Part 1's tokenizer and forward pass demo.
"""

import typer
import uvicorn
from rich.panel import Panel

from commands.common import console, create_header
from commands import generate, benchmarking, monitoring, testing

# Initialize main CLI app
app = typer.Typer(help="🚀 RuvonInference - Educational Inference Engine")

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
):
    """
    🌐 Start the HTTP API server

    This starts the FastAPI server with streaming text completion endpoints,
    demonstrating Part 4's HTTP server with /completions endpoint.
    """
    # Show header
    console.print(create_header())
    console.print()

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

    # Get attention implementation info
    try:
        from ruvoninference.attention import (
            get_available_implementations,
            recommend_implementation,
        )

        available_implementations = get_available_implementations()
        best_implementation = recommend_implementation(
            512, "cpu"
        )  # Typical server workload
        attention_info = f"⚡ Attention: [bold magenta]{best_implementation.value}[/bold magenta] (available: {', '.join([impl.value for impl in available_implementations])})\n"
    except ImportError:
        attention_info = "⚡ Attention: [bold yellow]Standard[/bold yellow] (FlashAttention not loaded)\n"

    # Show server info
    server_panel = Panel(
        f"🌐 Starting RuvonInference API Server\n"
        f"📍 Address: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"🔄 Auto-reload: [bold cyan]{'Enabled' if reload else 'Disabled'}[/bold cyan]\n"
        f"📦 Queue Mode: [bold {queue_style}]{queue_display}[/bold {queue_style}]\n"
        f"{attention_info}"
        f"📖 API Docs: [bold cyan]http://{host}:{port}/docs[/bold cyan]\n"
        f"🩺 Health Check: [bold cyan]http://{host}:{port}/health[/bold cyan]\n"
        f"🚀 ASGI Server: [bold yellow]uvicorn[/bold yellow] (uvloop, httptools, keep-alive)\n"
        f"📊 Config: [bold yellow]1 worker[/bold yellow] | [bold green]CORS enabled[/bold green] | [bold red]GZip disabled[/bold red]\n"
        f"📝 Logs: [bold cyan]logs/ruvoninference_*.log[/bold cyan]",
        title="🚀 RuvonInference API Server Configuration",
        style="green",
        border_style="green",
    )
    console.print(server_panel)
    console.print()

    # Instructions
    instructions_text = """
[bold blue]📋 Quick Test Commands:[/bold blue]

[bold yellow]1. Health Check (shows attention implementations):[/bold yellow]
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

[bold yellow]4. With specific attention implementation:[/bold yellow]
```bash
curl -X POST http://localhost:8000/completions \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello world", "max_tokens": 5, "attention_implementation": "sdpa"}'
```
    """

    console.print(Panel(instructions_text, style="blue", border_style="blue"))
    console.print()

    # Set environment variable for queue mode
    import os

    os.environ["QUEUE_MODE"] = queue_mode

    # Setup logging
    import logging
    from datetime import datetime

    # Create logs directory
    log_dir = "logs"
    import os

    os.makedirs(log_dir, exist_ok=True)

    # Setup file logging
    log_filename = (
        f"{log_dir}/ruvoninference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure logging
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Still show in terminal
        ],
    )

    # Start the server
    console.print("🚀 [bold green]Starting server...[/bold green]")
    console.print(f"📝 [bold cyan]Logs: {log_filename}[/bold cyan]")
    console.print()

    try:
        uvicorn.run(
            "ruvoninference.api.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            # Production optimizations
            loop="uvloop",  # Use uvloop for better async performance
            http="httptools",  # Use httptools for faster HTTP parsing
            ws="websockets",  # WebSocket support
            lifespan="on",  # Enable lifespan events
            access_log=False,  # Disable access logs for performance
            # Connection settings
            timeout_keep_alive=65,  # Server-side keep-alive timeout
            # Worker settings (single worker for GPU affinity)
            workers=1,  # One GPU = one process
            # Disable compression for streaming (gzip off for streams)
            # Note: FastAPI handles compression, disable at middleware level
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
    ℹ️  Show information about RuvonInference
    """
    console.print(create_header())
    console.print()

    info_text = """
[bold blue]🚀 RuvonInference - Educational Inference Engine[/bold blue]

A miniature but real inference system for transformer models, built from scratch
over 20 Parts. This project demonstrates modern LLM serving techniques including:

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
• ⚡ Continuous batching (Parts 6-8)
• 🚀 FlashAttention integration (Part 9) - Multiple attention implementations

[bold yellow]🎯 Upcoming Features:[/bold yellow]
• 📊 Logprobs API (Part 10)
• 📊 Telemetry and metrics (Part 11)
• 🔧 Custom Triton kernels (Part 12)
• 📄 Paged KV-cache allocator (Part 13)
• 🎭 Speculative decoding (Part 16)
• ⚖️  Quantization support (Part 17)

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

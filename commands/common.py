"""
Common utilities and shared functionality for CLI commands.

This module provides shared console setup, headers, server utilities,
and common patterns used across multiple command modules.
"""

import asyncio
import aiohttp
import subprocess
import signal
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


# Global console instance
console = Console()


def create_header() -> Panel:
    """
    Create the standard header panel for CLI commands.

    Returns:
        Panel: Rich panel with RuvonVLLM branding and emoji
    """
    header_text = Text()
    header_text.append("🚀 ", style="bold blue")
    header_text.append("RuvonVLLM", style="bold white")
    header_text.append(" - GPT-2 Inference Engine", style="bold cyan")

    return Panel(header_text, style="blue", border_style="blue", expand=False)


async def check_server(
    host: str = "127.0.0.1", port: int = 8000, timeout: int = 3
) -> bool:
    """
    Check if the API server is running and responsive.

    Args:
        host: Server hostname
        port: Server port
        timeout: Request timeout in seconds

    Returns:
        bool: True if server is running and responsive
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{host}:{port}/health", timeout=timeout
            ) as response:
                return response.status == 200
    except Exception:
        return False


async def start_server_if_needed(
    host: str = "127.0.0.1",
    port: int = 8000,
    auto_start: bool = True,
    wait_timeout: int = 30,
) -> tuple[bool, subprocess.Popen | None]:
    """
    Start the API server if needed and not already running.

    Args:
        host: Server hostname
        port: Server port
        auto_start: Whether to automatically start server if not running
        wait_timeout: How long to wait for server startup

    Returns:
        tuple: (success: bool, process: Popen | None)
    """
    is_running = await check_server(host, port)

    if is_running:
        console.print("✅ [bold green]Server is already running[/bold green]")
        console.print()
        return True, None

    if not auto_start:
        console.print(
            "❌ [bold red]Server not running. Use --auto-start to start automatically[/bold red]"
        )
        return False, None

    console.print("🚀 [bold blue]Starting API server...[/bold blue]")

    # Start server in background
    server_process = subprocess.Popen(
        ["make", "run-api"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # Create new process group for clean shutdown
    )

    # Wait for server to start
    for i in range(wait_timeout):
        await asyncio.sleep(1)
        if await check_server(host, port):
            console.print("✅ [bold green]Server started successfully[/bold green]")
            console.print()
            return True, server_process

    console.print("❌ [bold red]Failed to start server[/bold red]")
    return False, server_process


def cleanup_server_process(server_process: subprocess.Popen | None) -> None:
    """
    Clean up a server process, ensuring it's properly terminated.

    Args:
        server_process: The process to clean up, or None
    """
    if not server_process:
        return

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


async def get_queue_stats(
    session: aiohttp.ClientSession, host: str = "127.0.0.1", port: int = 8000
) -> dict | None:
    """
    Fetch current queue statistics from the server.

    Args:
        session: aiohttp session for making requests
        host: Server hostname
        port: Server port

    Returns:
        dict | None: Queue stats or None if failed
    """
    try:
        async with session.get(f"http://{host}:{port}/queue", timeout=2) as response:
            if response.status == 200:
                return await response.json()
            return None
    except Exception:
        return None


async def get_recent_completions(
    session: aiohttp.ClientSession,
    host: str = "127.0.0.1",
    port: int = 8000,
    limit: int = 20,
) -> list | None:
    """
    Fetch recent completions from the server.

    Args:
        session: aiohttp session for making requests
        host: Server hostname
        port: Server port
        limit: Maximum number of recent completions to fetch

    Returns:
        list | None: Recent completions or None if failed
    """
    try:
        async with session.get(
            f"http://{host}:{port}/queue/recent?limit={limit}", timeout=2
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
    except Exception:
        return None


def create_config_panel(title: str, config_items: dict, style: str = "blue") -> Panel:
    """
    Create a configuration panel with key-value pairs.

    Args:
        title: Panel title
        config_items: Dictionary of configuration key-value pairs
        style: Panel style

    Returns:
        Panel: Rich panel displaying configuration
    """
    config_text = ""
    for key, value in config_items.items():
        config_text += f"{key}: [bold cyan]{value}[/bold cyan]\n"

    return Panel(
        config_text.rstrip(),
        title=title,
        style=style,
        border_style=style,
    )


def create_stats_table(
    title: str, stats: dict, title_style: str = "bold green"
) -> Table:
    """
    Create a statistics table with key-value pairs.

    Args:
        title: Table title
        stats: Dictionary of statistics
        title_style: Title style

    Returns:
        Table: Rich table displaying statistics
    """
    table = Table(title=title, show_header=True, header_style=title_style)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow")

    for key, value in stats.items():
        table.add_row(key, str(value))

    return table


class RequestTracker:
    """
    Utility class for tracking request timing and results.

    This helps with monitoring request performance and collecting
    statistics across different command implementations.
    """

    def __init__(self):
        self.start_time = time.time()
        self.requests_sent = 0
        self.requests_completed = 0
        self.requests_failed = 0
        self.total_response_time = 0.0

    def record_request_sent(self):
        """Record that a request was sent."""
        self.requests_sent += 1

    def record_request_completed(self, response_time: float):
        """Record that a request completed successfully."""
        self.requests_completed += 1
        self.total_response_time += response_time

    def record_request_failed(self):
        """Record that a request failed."""
        self.requests_failed += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.requests_sent == 0:
            return 0.0
        return (self.requests_completed / self.requests_sent) * 100

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.requests_completed == 0:
            return 0.0
        return self.total_response_time / self.requests_completed

    @property
    def elapsed_time(self) -> float:
        """Get total elapsed time since tracking started."""
        return time.time() - self.start_time

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.requests_sent / elapsed

    def get_summary_stats(self) -> dict:
        """Get summary statistics as a dictionary."""
        return {
            "Total Requests": self.requests_sent,
            "Successful": self.requests_completed,
            "Failed": self.requests_failed,
            "Success Rate": f"{self.success_rate:.1f}%",
            "Avg Response Time": f"{self.average_response_time:.2f}s",
            "Requests/Second": f"{self.requests_per_second:.1f}",
            "Total Time": f"{self.elapsed_time:.2f}s",
        }

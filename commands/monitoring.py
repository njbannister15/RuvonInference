"""
Real-time monitoring commands for RuvonVLLM.

This module provides live dashboard functionality for monitoring
queue statistics, active requests, and server health in real-time.
"""

import typer
import asyncio
import aiohttp
import time
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.table import Table
from rich.panel import Panel

from .common import (
    console,
    create_header,
    create_config_panel,
    start_server_if_needed,
    cleanup_server_process,
    get_queue_stats,
    get_recent_completions,
)

app = typer.Typer(help="🔍 Real-time monitoring commands")


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
    # Show header
    console.print(create_header())
    console.print()

    # Show monitor configuration
    config_items = {
        "🌐 Server": f"http://{host}:{port}",
        "🔄 Refresh rate": f"{refresh_rate}s",
        "🚀 Auto-start server": "Yes" if auto_start_server else "No",
        "🧪 Send test requests": "Yes" if send_test_requests else "No",
    }
    config_panel = create_config_panel("📊 Live Dashboard Setup", config_items)
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
        uptime_str = f"{int(uptime // 3600):02d}:{int((uptime % 3600) // 60):02d}:{int(uptime % 60):02d}"

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

        async def get_health_data():
            """Get health data from server."""
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

        async def run_monitor():
            # Start server if needed
            success, process = await start_server_if_needed(
                host, port, auto_start_server
            )
            nonlocal server_process
            server_process = process

            if not success:
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
                            stats = await get_queue_stats(session, host, port)
                            health = await get_health_data()
                            server_completions = await get_recent_completions(
                                session, host, port, 10
                            )

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
        cleanup_server_process(server_process)


if __name__ == "__main__":
    app()

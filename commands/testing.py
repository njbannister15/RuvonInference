"""
Testing and load testing commands for RuvonInference.

This module provides stress testing and rapid testing functionality
for evaluating the queue system and server performance under load.
"""

import typer
import asyncio
import aiohttp
import time
from rich.live import Live
from rich.columns import Columns
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .common import (
    console,
    create_header,
    create_config_panel,
    start_server_if_needed,
    cleanup_server_process,
    get_queue_stats,
)

app = typer.Typer(help="🧪 Testing and load testing commands")


@app.command()
def stress_test(
    max_requests: int = typer.Option(
        100, "--max-requests", "-n", help="Total number of requests to send"
    ),
    burst_size: int = typer.Option(
        10,
        "--burst-size",
        "-b",
        help="Starting burst size - number of rapid requests per wave (10, 20, 30, etc.)",
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
    🚀 Run stress test with increasing burst sizes

    This sends requests in increasing bursts (10, 20, 30, 40...) to test
    the queue system's ability to handle multiple concurrent requests.
    """
    # Show header
    console.print(create_header())
    console.print()

    # Show test configuration
    config_items = {
        "📊 Total requests": str(max_requests),
        "📦 Burst size": f"{burst_size} (increments by {burst_size})",
        "🌐 Server": f"http://{host}:{port}",
        "📝 Prompt": f"'{prompt}'",
        "🎭 Tokens per request": str(max_tokens),
        "🔄 Auto-start server": "Yes" if auto_start_server else "No",
    }
    config_panel = create_config_panel("🧪 Stress Test Setup", config_items)
    console.print(config_panel)
    console.print()

    server_process = None

    try:
        # Send a batch of requests
        async def send_burst(session, burst_size, burst_num):
            """Send a burst of requests concurrently and measure timing."""
            request_data = {
                "prompt": f"{prompt} (burst {burst_num})",
                "max_tokens": max_tokens,
            }

            start_time = time.time()
            tasks = []

            for i in range(burst_size):
                task = session.post(
                    f"http://{host}:{port}/completions",
                    json=request_data,
                    timeout=300,  # 5 minute timeout per request
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
                    "burst_size": burst_size,
                    "burst_num": burst_num,
                    "successful": successful,
                    "failed": failed,
                    "total_time": end_time - start_time,
                    "requests_per_second": burst_size / (end_time - start_time)
                    if end_time > start_time
                    else 0,
                }

            except Exception as e:
                return {
                    "burst_size": burst_size,
                    "burst_num": burst_num,
                    "successful": 0,
                    "failed": burst_size,
                    "total_time": time.time() - start_time,
                    "requests_per_second": 0,
                    "error": str(e),
                }

        async def run_stress_test():
            # Start server if needed
            success, process = await start_server_if_needed(
                host, port, auto_start_server
            )
            nonlocal server_process
            server_process = process

            if not success:
                return

            # Initialize session
            async with aiohttp.ClientSession() as session:
                # Get initial queue stats
                try:
                    initial_stats = await get_queue_stats(session, host, port)
                    if initial_stats:
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

                current_burst_size = burst_size
                burst_num = 1

                console.print("🚀 [bold blue]Starting stress test...[/bold blue]")
                console.print()

                # Create results table
                results_table = Table(
                    title="📊 Real-time Stress Test Results",
                    show_header=True,
                    header_style="bold magenta",
                )
                results_table.add_column("Burst #", style="cyan", no_wrap=True)
                results_table.add_column("Size", style="yellow")
                results_table.add_column("Success", style="green")
                results_table.add_column("Failed", style="red")
                results_table.add_column("Time (s)", style="blue")
                results_table.add_column("Req/s", style="magenta")

                with Live(results_table, refresh_per_second=1) as live:
                    while total_sent < max_requests:
                        # Adjust burst size to not exceed max_requests
                        remaining = max_requests - total_sent
                        actual_burst_size = min(current_burst_size, remaining)

                        # Send burst
                        result = await send_burst(session, actual_burst_size, burst_num)
                        results.append(result)

                        # Update stats
                        total_sent += actual_burst_size
                        total_successful += result["successful"]
                        total_failed += result["failed"]

                        # Add to table
                        results_table.add_row(
                            str(burst_num),
                            str(actual_burst_size),
                            str(result["successful"]),
                            str(result["failed"]),
                            f"{result['total_time']:.2f}",
                            f"{result['requests_per_second']:.1f}",
                        )

                        # Update live display
                        live.update(results_table)

                        # Move to next burst
                        current_burst_size += burst_size
                        burst_num += 1

                        # Small delay between bursts
                        await asyncio.sleep(1)

                console.print()

                # Get final queue stats
                try:
                    final_stats = await get_queue_stats(session, host, port)
                    if final_stats:
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
                    f"{(total_successful / total_sent * 100):.1f}%"
                    if total_sent > 0
                    else "0%",
                )
                summary_table.add_row("Total Bursts", str(len(results)))
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
        cleanup_server_process(server_process)


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
    # Show header
    console.print(create_header())
    console.print()

    # Show test configuration
    config_items = {
        "📊 Total requests": str(total_requests),
        "⚡ Concurrent limit": str(concurrent_limit),
        "🌐 Server": f"http://{host}:{port}",
        "📝 Prompt": f"'{prompt}'",
        "🎭 Tokens per request": str(max_tokens),
        "🔄 Auto-start server": "Yes" if auto_start_server else "No",
        "📊 Monitor queue": "Yes" if monitor_queue else "No",
    }
    config_panel = create_config_panel("⚡ Rapid Queue Test Setup", config_items)
    console.print(config_panel)
    console.print()

    server_process = None
    start_time = time.time()

    try:

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
            success, process = await start_server_if_needed(
                host, port, auto_start_server
            )
            nonlocal server_process
            server_process = process

            if not success:
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
                            current_stats = await get_queue_stats(session, host, port)

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
                    "Success Rate", f"{(successful / total_requests * 100):.1f}%"
                )
                final_table.add_row("Total Time", f"{total_time:.2f}s")
                final_table.add_row(
                    "Requests/Second", f"{total_requests / total_time:.1f}"
                )

                console.print(final_table)
                console.print()

                # Get final queue stats
                try:
                    final_stats = await get_queue_stats(session, host, port)
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
        cleanup_server_process(server_process)


if __name__ == "__main__":
    app()

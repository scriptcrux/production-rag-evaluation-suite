"""Interactive terminal CLI for FinSight Enterprise Financial Analyst."""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.financial_analyst import FinancialAnalystAgent
from src.domain.models import UserQuery

console = Console()


async def run_interactive_cli():
    """Runs interactive REPL for FinSight."""
    console.print(
        Panel.fit(
            "[bold cyan]FinSight Enterprise Financial Analyst & SEC 10-K Compliance Agent[/bold cyan]\n"
            "[dim]Powered by Hybrid RAG (BM25 + Dense RRF), Multi-Turn Memory & DeepEval Governance[/dim]",
            border_style="cyan",
        )
    )

    agent = FinancialAnalystAgent()
    with console.status("[bold green]Indexing knowledge base filings...[/bold green]"):
        await agent.initialize()

    session_id = "interactive_cli_session"
    console.print("[green]Ready! Ask financial questions about Microsoft, NVIDIA, or Apple FY2024 Form 10-Ks.[/green]")
    console.print("[dim]Type 'exit' or 'quit' to terminate.[/dim]\n")

    while True:
        try:
            user_input = input("\n[Analyst Query] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Exiting FinSight. Goodbye![/yellow]")
                break

            with console.status("[bold yellow]Synthesizing audited disclosures...[/bold yellow]"):
                query = UserQuery(text=user_input, session_id=session_id)
                response = await agent.analyze(query)

            # Display Executive Summary
            console.print("\n[bold green]Executive Summary:[/bold green]")
            console.print(response.executive_summary)

            # Display Extracted Metrics Table
            if response.financial_metrics:
                table = Table(title="Audited Financial Metrics", show_header=True, header_style="bold magenta")
                table.add_column("Metric", style="dim")
                table.add_column("Value", style="bold")
                table.add_column("Period")
                table.add_column("YoY Change", style="cyan")
                table.add_column("Status", style="green")

                for m in response.financial_metrics:
                    table.add_row(
                        m.metric_name,
                        m.value,
                        m.period,
                        m.yoy_change or "N/A",
                        m.audit_status,
                    )
                console.print(table)

            # Display Identified Risks
            if response.risk_assessment:
                console.print("\n[bold red]Material Risk Disclosures:[/bold red]")
                for r in response.risk_assessment:
                    console.print(f" • [dim]{r}[/dim]")

            # Display Citations
            if response.citations:
                console.print("\n[bold blue]Source Citations:[/bold blue]")
                for c in response.citations:
                    console.print(f" • [bold]{c.ticker}[/bold] | [{c.chunk_id}] {c.section}")

            # Telemetry footer
            if response.telemetry:
                console.print(
                    f"\n[dim]Latency: {response.telemetry.total_latency_ms:.1f}ms | "
                    f"Tokens: {response.telemetry.total_tokens} | "
                    f"Cost: ${response.telemetry.estimated_cost_usd:.5f} | "
                    f"Confidence: {response.confidence_score * 100:.1f}%[/dim]"
                )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(run_interactive_cli())


import typer

app = typer.Typer(
    help="Benchmark an LLM served over an OpenAI-compatible streaming API.",
    add_completion=False,  # hide the shell-completion options from --help
)


@app.command()
def main(
    model: str = typer.Option(..., help="Model name, e.g. llama3.2:3b"),
) -> None:
    """Benchmark a model."""
    print(f"llm-bench is installed. Model: {model}")
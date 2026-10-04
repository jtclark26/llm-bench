import typer

app = typer.Typer(
    help="Benchmark an LLM served over an OpenAI-compatible streaming API.",
    add_completion=False,  # hides the --install-completion options to keep --help short
)


@app.command()
def main(
    model: str = typer.Option(..., help="Model name, e.g. llama3.2:3b"),
) -> None:
    """Placeholder command so we can check the install works. Filled in at step 5."""
    print(f"llm-bench is installed. You asked for model: {model}")

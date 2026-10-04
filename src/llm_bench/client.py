"""Send a single streaming chat request and time the response.

send_request only records when each line of the stream arrives. Parsing and
the calculations happen afterwards in summarise_stream, which keeps them
testable without a server.
"""

import json
import time
from dataclasses import dataclass

import httpx

DEFAULT_PROMPT = "Write a short story (about 300 words) about a robot learning to paint."


@dataclass
class RequestResult:
    """Outcome of one request. All times are in seconds."""

    ok: bool
    ttft: float | None = None  # time to first token
    latency: float | None = None  # request sent until stream finished
    output_tokens: int = 0
    tokens_per_sec: float | None = None  # generation speed after the first token
    usage_reported: bool = False  # token count came from the server, not estimated
    error: str | None = None


def parse_sse_line(line: str) -> str | None:
    """Return the payload of an SSE 'data:' line, or None for any other line."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    return line[len("data:"):].strip()


def chunk_text(chunk: dict) -> str:
    """Return the text content of a stream chunk, or '' if it has none."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def summarise_stream(start_time: float, timed_lines: list[tuple[float, str]]) -> RequestResult:
    """Calculate the metrics for one request from (arrival_time, line) pairs."""
    first_token_time = None
    last_token_time = None
    chunk_count = 0
    usage_tokens = None

    for arrival_time, line in timed_lines:
        payload = parse_sse_line(line)
        if payload is None:
            continue
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)

        if "error" in chunk:  # some servers report errors mid-stream
            return RequestResult(ok=False, error=str(chunk["error"]))

        if chunk.get("usage"):  # only in the final chunk, if supported
            usage_tokens = chunk["usage"].get("completion_tokens")

        if chunk_text(chunk):
            chunk_count += 1
            if first_token_time is None:
                first_token_time = arrival_time
            last_token_time = arrival_time

    if first_token_time is None:
        return RequestResult(ok=False, error="stream ended without any tokens")

    # Use the server's token count if it sent one. Otherwise count text chunks,
    # which is close because Ollama sends about one token per chunk.
    if usage_tokens:
        output_tokens = usage_tokens
        usage_reported = True
    else:
        output_tokens = chunk_count
        usage_reported = False

    # The first token is already covered by TTFT, so measure the n - 1 gaps
    # between n tokens.
    generation_time = last_token_time - first_token_time
    if output_tokens > 1 and generation_time > 0:
        tokens_per_sec = (output_tokens - 1) / generation_time
    else:
        tokens_per_sec = None

    stream_end_time = timed_lines[-1][0]

    return RequestResult(
        ok=True,
        ttft=first_token_time - start_time,
        latency=stream_end_time - start_time,
        output_tokens=output_tokens,
        tokens_per_sec=tokens_per_sec,
        usage_reported=usage_reported,
    )


def send_request(
    base_url: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 256,
    timeout: float = 120.0,
) -> RequestResult:
    """Send one streaming chat request and return its timings.

    Errors are returned as a failed RequestResult instead of raised, so one
    bad request doesn't stop a benchmark run.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},  # request a final usage chunk
    }

    timed_lines = []
    start_time = time.perf_counter()
    try:
        with httpx.stream("POST", url, json=body, timeout=timeout) as response:
            if response.status_code != 200:
                response.read()  # load the body so the error message can be shown
                return RequestResult(
                    ok=False, error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
            for line in response.iter_lines():
                timed_lines.append((time.perf_counter(), line))
        return summarise_stream(start_time, timed_lines)
    except httpx.HTTPError as e:  # connection refused, timeout, dropped stream
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")
    except ValueError as e:  # invalid JSON in the stream
        return RequestResult(ok=False, error=f"bad data in stream: {e}")
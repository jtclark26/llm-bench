"""Send one streaming chat request and measure how long each part takes.

The network code (send_request) only records *when* each line of the stream
arrived. All the maths is done afterwards in pure functions (parse_sse_line,
summarise_stream), so they can be tested with fake data and no server.
"""

import json
import time
from dataclasses import dataclass

import httpx

DEFAULT_PROMPT = "Write a short story (about 300 words) about a robot learning to paint."


@dataclass
class RequestResult:
    """The outcome of one request. Times are in seconds."""

    ok: bool
    ttft: float | None = None  # time to first token
    latency: float | None = None  # end-to-end: request sent -> stream finished
    output_tokens: int = 0
    tokens_per_sec: float | None = None  # generation speed after the first token
    usage_reported: bool = False  # True if the server told us the token count
    error: str | None = None


def parse_sse_line(line: str) -> str | None:
    """Return the data part of one server-sent events (SSE) line, or None.

    An SSE stream is plain text. Each event looks like 'data: <something>'
    followed by a blank line. Blank lines and anything else carry no data
    for us, so we return None for those.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    return line[len("data:"):].strip()


def chunk_text(chunk: dict) -> str:
    """Get the generated text out of one parsed chunk ('' if there is none)."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def summarise_stream(start_time: float, timed_lines: list[tuple[float, str]]) -> RequestResult:
    """Turn the raw (arrival_time, line) pairs from one stream into a result.

    start_time is when the request was sent. Times come from time.perf_counter().
    """
    first_token_time = None
    last_token_time = None
    chunk_count = 0  # chunks that contained some text
    usage_tokens = None

    for arrival_time, line in timed_lines:
        payload = parse_sse_line(line)
        if payload is None:
            continue
        if payload == "[DONE]":  # the server's "end of stream" marker
            break

        chunk = json.loads(payload)

        if "error" in chunk:  # some servers report errors inside the stream
            return RequestResult(ok=False, error=str(chunk["error"]))

        if chunk.get("usage"):  # only sent at the end, if the server supports it
            usage_tokens = chunk["usage"].get("completion_tokens")

        if chunk_text(chunk):
            chunk_count += 1
            if first_token_time is None:
                first_token_time = arrival_time
            last_token_time = arrival_time

    if first_token_time is None:
        return RequestResult(ok=False, error="stream ended without any tokens")

    # Prefer the server's real token count; fall back to counting text chunks
    # (Ollama sends roughly one token per chunk, so this is a close estimate).
    if usage_tokens:
        output_tokens = usage_tokens
        usage_reported = True
    else:
        output_tokens = chunk_count
        usage_reported = False

    # Speed of generation *after* the first token. The first token's time is
    # already in TTFT, so we count the gaps between tokens: n tokens -> n - 1 gaps.
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
    """Send one streaming chat request and time every line that comes back.

    Never raises for network or server problems: it returns ok=False instead,
    so one failed request can't crash a whole benchmark run.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        # Ask the server to send the exact token count in a final chunk.
        "stream_options": {"include_usage": True},
    }

    timed_lines = []
    start_time = time.perf_counter()
    try:
        with httpx.stream("POST", url, json=body, timeout=timeout) as response:
            if response.status_code != 200:
                response.read()  # load the error body so we can show it
                return RequestResult(
                    ok=False, error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
            for line in response.iter_lines():
                timed_lines.append((time.perf_counter(), line))
        return summarise_stream(start_time, timed_lines)
    except httpx.HTTPError as e:  # connection refused, timeout, dropped stream...
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")
    except ValueError as e:  # a chunk that wasn't valid JSON
        return RequestResult(ok=False, error=f"bad data in stream: {e}")

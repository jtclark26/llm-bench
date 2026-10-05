"""Run requests concurrently using a thread pool."""

import time
from concurrent.futures import ThreadPoolExecutor

from llm_bench.client import RequestResult, send_request


def warm_up(base_url: str, model: str) -> RequestResult:
    """Send one unmeasured request so model loading time isn't counted.

    Also confirms the server and model are reachable before the real run.
    """
    return send_request(base_url, model, prompt="Say hello.", max_tokens=5)


def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
) -> tuple[list[RequestResult], float]:
    """Send num_requests requests with up to `concurrency` in flight at once.

    Returns the results and the total wall-clock time in seconds.
    """
    start_time = time.perf_counter()

    # Each worker takes the next queued request when it finishes one, so at
    # most `concurrency` requests run at the same time.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for _ in range(num_requests):
            future = pool.submit(send_request, base_url, model, prompt, max_tokens)
            futures.append(future)

        results = []
        for future in futures:
            try:
                results.append(future.result())  # blocks until this request finishes
            except Exception as e:
                # send_request already handles network errors; this catches
                # anything unexpected so one bug doesn't lose the whole run.
                results.append(RequestResult(ok=False, error=f"unexpected error: {e!r}"))

    wall_time = time.perf_counter() - start_time
    return results, wall_time

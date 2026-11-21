"""Minimal OpenAI-compatible chat client for text-only uncertainty methods.

Works with any server that implements ``POST {base_url}/chat/completions``:
OpenAI, vLLM, Ollama (``http://localhost:11434/v1``), LM Studio, llama.cpp
server, OpenRouter, Together, Groq, ...

Only the Python standard library is used, so this backend adds no dependency.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

RETRY_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class APIError(RuntimeError):
    """Raised when the API returns an error that is not worth retrying."""


@dataclass
class UsageCounter:
    """Accumulates request and token counts so the cost column can be filled.

    Attributes:
        requests: Number of HTTP requests sent.
        prompt_tokens: Prompt tokens reported by the server.
        completion_tokens: Completion tokens reported by the server.
        seconds: Wall-clock time spent waiting for the server.
    """

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> dict[str, float]:
        """Return the current counts as a plain dict.

        Returns:
            A copy of the counters.
        """
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 4),
        }


@dataclass
class OpenAICompatibleClient:
    """Chat-completions client for a model served behind an OpenAI-compatible API.

    Args:
        model: Model name as the server knows it (e.g. ``"qwen2.5:0.5b"``).
        base_url: API root including ``/v1``. Defaults to ``$OPENAI_BASE_URL``
            or ``https://api.openai.com/v1``.
        api_key: Bearer token. Defaults to ``$OPENAI_API_KEY``; local servers
            usually accept any value.
        timeout: Seconds to wait for one request.
        max_retries: Retries for rate limits, timeouts and 5xx responses.
        backoff: Base seconds for exponential backoff between retries.
        supports_n: Whether the server honours ``n > 1``. ``None`` means
            "detect on first use": if fewer choices come back than requested,
            the client switches to one request per sample.

    Example:
        >>> client = OpenAICompatibleClient(model="qwen2.5:0.5b",
        ...                                 base_url="http://localhost:11434/v1")
        >>> client.chat([{"role": "user", "content": "Capital of France?"}])  # doctest: +SKIP
        ['Paris']
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 4
    backoff: float = 1.0
    supports_n: bool | None = None
    usage: UsageCounter = field(default_factory=UsageCounter)

    def __post_init__(self) -> None:
        """Fill defaults from the environment."""
        self.base_url = (
            self.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one chat-completions request with retries.

        Args:
            payload: JSON body without the ``model`` field.

        Returns:
            The decoded JSON response.

        Raises:
            APIError: On a non-retryable error or when retries are exhausted.
        """
        body = json.dumps({"model": self.model, **payload}).encode()
        request = urllib.request.Request(  # noqa: S310 - URL is user-configured
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error = ""
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    data = json.loads(response.read().decode())
                self._record(data, time.perf_counter() - start)
                return data
            except urllib.error.HTTPError as err:
                self.usage.seconds += time.perf_counter() - start
                detail = err.read().decode(errors="replace")[:500]
                last_error = f"HTTP {err.code}: {detail}"
                if err.code not in RETRY_STATUS_CODES:
                    raise APIError(last_error) from err
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                self.usage.seconds += time.perf_counter() - start
                last_error = f"connection error: {err}"
            if attempt < self.max_retries:
                time.sleep(self.backoff * 2**attempt)
        msg = f"giving up after {self.max_retries + 1} attempts ({last_error})"
        raise APIError(msg)

    def _record(self, data: dict[str, Any], seconds: float) -> None:
        """Add one response to the usage counters.

        Args:
            data: Decoded response.
            seconds: Time the request took.
        """
        usage = data.get("usage") or {}
        self.usage.requests += 1
        self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.usage.seconds += seconds

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 64,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Return ``n`` completions for a chat prompt.

        If the server ignores ``n`` (Ollama, some proxies), the missing
        completions are fetched with extra single requests, each with its own
        seed so the samples differ.

        Args:
            messages: OpenAI chat messages.
            n: Number of completions.
            temperature: Sampling temperature; ``0`` gives the greedy answer.
            max_tokens: Maximum new tokens per completion.
            seed: Optional base seed for reproducible sampling.
            stop: Optional stop sequences.

        Returns:
            ``n`` completion strings, stripped of surrounding whitespace.

        Raises:
            ValueError: If ``n`` is smaller than 1.
        """
        if n < 1:
            msg = f"n must be >= 1, got {n}"
            raise ValueError(msg)
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        outputs: list[str] = []
        if n > 1 and self.supports_n is not False:
            data = self._post({**payload, "n": n})
            outputs = _contents(data)
            if self.supports_n is None:
                self.supports_n = len(outputs) >= n
        while len(outputs) < n:
            single = dict(payload)
            if seed is not None:
                single["seed"] = seed + len(outputs)
            outputs.extend(_contents(self._post(single)))
        return outputs[:n]


def _contents(data: dict[str, Any]) -> list[str]:
    """Extract message texts from a chat-completions response.

    Args:
        data: Decoded response.

    Returns:
        One string per choice (empty string if the server sent no content).

    Raises:
        APIError: If the response has no ``choices`` list.
    """
    choices = data.get("choices")
    if not isinstance(choices, list):
        msg = f"response has no 'choices': {str(data)[:300]}"
        raise APIError(msg)
    return [((c.get("message") or {}).get("content") or "").strip() for c in choices]

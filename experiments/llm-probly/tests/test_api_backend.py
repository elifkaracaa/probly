"""Tests for the text-only API backend (F2): client, prompts/parsers, plugins, generator.

All tests use a local mock OpenAI-compatible server: no network, no model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import TYPE_CHECKING, Any

from common import text_only
from common.openai_client import APIError, OpenAICompatibleClient
from common.text_only import (
    PT_PROMPT,
    client_from_rows,
    empirical_p_true,
    is_false,
    is_true,
    parse_confidence,
    verbalized_2s,
)
from methods import p_true_empirical_api, verbalized_2s_api
import pytest

from scripts import generate_api, score_api_cache

if TYPE_CHECKING:
    from pathlib import Path

Responder = Callable[[list[dict[str, str]], int, dict[str, Any]], str]


class MockServer:
    """Mock ``/v1/chat/completions`` endpoint running in a background thread."""

    def __init__(
        self,
        responder: Responder,
        *,
        honour_n: bool = True,
        fail_first: int = 0,
        fail_status: int = 429,
    ):
        self.responder, self.honour_n = responder, honour_n
        self.fail_first, self.fail_status = fail_first, fail_status
        self.requests: list[dict[str, Any]] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append(payload)
                if len(mock.requests) <= mock.fail_first:
                    return self._send(mock.fail_status, {"error": {"message": "mock failure"}})
                n = int(payload.get("n", 1)) if mock.honour_n else 1
                choices = [
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": mock.responder(payload["messages"], i, payload),
                        },
                    }
                    for i in range(n)
                ]
                return self._send(
                    200,
                    {
                        "choices": choices,
                        "usage": {"prompt_tokens": 10, "completion_tokens": n},
                    },
                )

            def _send(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (name from BaseHTTPRequestHandler)
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def quiz(messages: list[dict[str, str]], index: int, payload: dict[str, Any]) -> str:
    """Fake model: sure about France, unsure about the hard question."""
    first, last = messages[0]["content"], messages[-1]["content"]
    hard = "hardest" in first
    if last.startswith("How confident"):
        return "20" if hard else "95"
    if last.endswith("The possible answer is:"):
        return ("(B)" if index % 2 else "A") if hard else "(A) True"
    if hard:
        return "Zork" if payload["temperature"] == 0 else ["Zork", "Blip", "Quux"][(payload.get("seed", 0) + index) % 3]
    return "Paris"


@pytest.fixture
def server() -> Iterator[MockServer]:
    s = MockServer(quiz)
    yield s
    s.close()


def client_for(s: MockServer, max_retries: int = 4) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(model="mock", base_url=s.base_url, backoff=0.0, max_retries=max_retries)


# ---------------- client ----------------


def test_chat_uses_n_in_one_request(server: MockServer) -> None:
    client = client_for(server)
    assert client.chat([{"role": "user", "content": "x"}], n=5, temperature=1.0) == ["Paris"] * 5
    assert len(server.requests) == 1
    assert server.requests[0]["n"] == 5
    assert client.supports_n is True
    assert client.usage.snapshot()["completion_tokens"] == 5


def test_chat_falls_back_when_server_ignores_n() -> None:
    s = MockServer(quiz, honour_n=False)
    try:
        client = client_for(s)
        out = client.chat([{"role": "user", "content": "hardest"}], n=4, temperature=1.0, seed=0)
        client.chat([{"role": "user", "content": "x"}], n=2, temperature=1.0)
    finally:
        s.close()
    assert len(out) == 4
    assert len(set(out)) > 1
    assert client.supports_n is False
    assert [r.get("seed") for r in s.requests[1:4]] == [1, 2, 3]
    assert len(s.requests) == 6  # second call no longer tries n


@pytest.mark.parametrize(
    ("fail_first", "status", "max_retries", "ok"),
    [(2, 429, 3, True), (9, 503, 1, False)],
)
def test_chat_retries(fail_first: int, status: int, max_retries: int, ok: bool) -> None:
    s = MockServer(quiz, fail_first=fail_first, fail_status=status)
    try:
        client = client_for(s, max_retries=max_retries)
        if ok:
            assert client.chat([{"role": "user", "content": "x"}]) == ["Paris"]
        else:
            with pytest.raises(APIError, match="giving up"):
                client.chat([{"role": "user", "content": "x"}])
    finally:
        s.close()


def test_chat_does_not_retry_auth_error() -> None:
    s = MockServer(quiz, fail_first=5, fail_status=401)
    try:
        with pytest.raises(APIError, match="HTTP 401"):
            client_for(s, max_retries=3).chat([{"role": "user", "content": "x"}])
    finally:
        s.close()
    assert len(s.requests) == 1


def test_chat_rejects_bad_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        OpenAICompatibleClient(model="m", base_url="http://127.0.0.1:1/v1").chat([], n=0)


def test_client_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAICompatibleClient(model="m")
    assert (client.base_url, client.api_key) == ("http://example.test/v1", "sk-test")


# ---------------- prompts and parsers (must match scripts/r3_methods.py) ----------------


def test_prompts_match_week1() -> None:
    r3 = pytest.importorskip("scripts.r3_methods")  # needs torch; skipped where it is missing
    assert r3.PT_PROMPT == text_only.PT_PROMPT
    assert r3.ASK == text_only.ASK
    for reply in ["A", "(B)", "true", "False.", "maybe", "a)", "Absolutely"]:
        assert (r3.is_true(reply), r3.is_false(reply)) == (
            is_true(reply),
            is_false(reply),
        )
    for reply in ["95", "0.9", "7.5", "none", "150"]:
        assert r3.parse_confidence(reply, 100) == parse_confidence(reply, 100)


@pytest.mark.parametrize(
    ("reply", "t", "f"),
    [("A", True, False), ("(A) True", True, False), ("true", True, False), ("B", False, True), ("(b)", False, True),
     ("False", False, True), ("Absolutely", False, False), ("maybe", False, False)],
)  # fmt: skip
def test_true_false_parsing(reply: str, t: bool, f: bool) -> None:
    assert (is_true(reply), is_false(reply)) == (t, f)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("95", 0.95),
        ("0.9", 0.9),
        ("7.5", 0.075),
        ("150", 1.0),
        ("I am 80% sure", 0.8),
        ("none", None),
    ],
)
def test_parse_confidence(reply: str, expected: float | None) -> None:
    assert parse_confidence(reply, 100) == (None if expected is None else pytest.approx(expected))


def test_api_reflexive_methods(server: MockServer) -> None:
    client = client_for(server)
    p, texts, unparsed = empirical_p_true(client, "Capital of France?", "Paris", n=10)
    assert (p, len(texts), unparsed) == (1.0, 10, 0)
    assert server.requests[-1]["messages"][0]["content"] == PT_PROMPT.format(q="Capital of France?", a="Paris")
    assert server.requests[-1]["max_tokens"] == 3
    p_hard, _, _ = empirical_p_true(client, "the hardest question", "Zork", n=10)
    assert p_hard == pytest.approx(0.5)
    assert verbalized_2s(client, "Capital of France?", "Paris") == (0.95, False, "95")
    chat = server.requests[-1]["messages"]
    assert [m["role"] for m in chat] == ["user", "assistant", "user"]
    assert chat[1]["content"] == "Paris"


def test_verbalized_unparsable() -> None:
    s = MockServer(lambda *_: "no idea")
    try:
        assert verbalized_2s(client_for(s), "q", "a") == (0.0, True, "no idea")
    finally:
        s.close()


def test_client_from_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"api": {"model": "m", "base_url": "http://a/v1"}}]
    assert client_from_rows(rows).base_url == "http://a/v1"
    monkeypatch.setenv("LLM_PROBLY_API_BASE_URL", "http://b/v1")
    assert client_from_rows(rows).base_url == "http://b/v1"
    with pytest.raises(ValueError, match="no 'api' field"):
        client_from_rows([{"model": "Qwen/Qwen2.5-0.5B-Instruct"}])


# ---------------- generator + plugins end to end ----------------


@pytest.fixture
def api_cache(server: MockServer, tmp_path: Path) -> list[dict]:
    source = tmp_path / "triviaqa_3.jsonl"
    questions = [
        ("Capital of France?", ["Paris"]),
        ("the hardest question", ["Blip"]),
        ("France capital again?", ["Paris"]),
    ]
    source.write_text(
        "".join(
            json.dumps({"id": i, "dataset": "triviaqa", "question": q, "aliases": a}) + "\n"
            for i, (q, a) in enumerate(questions)
        )
    )
    out = generate_api.main([
        "--model-key", "mock-model", "--api-model", "mock", "--base-url", server.base_url,
        "--source", str(source), "--out-root", str(tmp_path / "data"), "--num-samples", "6",
    ])  # fmt: skip
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_generate_api_writes_cache_format(api_cache: list[dict], tmp_path: Path) -> None:
    row = api_cache[0]
    for key in (
        "id",
        "dataset",
        "question",
        "aliases",
        "greedy_answer",
        "samples",
        "model",
        "correct",
        "samples_correct",
    ):
        assert key in row
    assert row["model"] == "api:mock"
    assert row["access"] == "text"
    assert row["samples_ll_mean"] is None
    assert len(row["samples"]) == 6
    assert [r["correct"] for r in api_cache] == [True, False, True]
    meta = json.loads((tmp_path / "data" / "mock-model" / "triviaqa_3.meta.json").read_text())
    assert meta["n"] == 3
    assert meta["accuracy"] == pytest.approx(2 / 3)


def test_api_plugins_on_generated_cache(api_cache: list[dict]) -> None:
    for plugin in (p_true_empirical_api, verbalized_2s_api):
        assert plugin.NEEDS_MODEL is False
        assert plugin.FAMILY == "self_eval"
        record = plugin.score(api_cache)
        assert record["method"] == plugin.NAME
        assert record["ids"] == [0, 1, 2]
        assert record["failed"] == [False, False, False]
        unc = record["uncertainty"]
        # the hard question is the most uncertain
        assert unc[1] > unc[0], plugin.NAME
        assert unc[1] > unc[2], plugin.NAME
        assert record["api_requests_per_question"] >= 1


def test_api_plugins_refuse_local_cache() -> None:
    rows = [
        {
            "id": 0,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "question": "q",
            "greedy_answer": "a",
        }
    ]
    for plugin in (p_true_empirical_api, verbalized_2s_api):
        with pytest.raises(ValueError, match=r"generate_api\.py"):
            plugin.score(rows)


def test_generate_api_resume_after_crash(server: MockServer, tmp_path: Path) -> None:
    source = tmp_path / "triviaqa_3.jsonl"
    source.write_text(
        "".join(
            json.dumps({"id": i, "dataset": "triviaqa", "question": q, "aliases": ["Paris"]}) + "\n"
            for i, q in enumerate(["Capital of France?", "France capital?", "Paris is the capital of?"])
        )
    )
    args = ["--model-key", "m", "--api-model", "mock", "--base-url", server.base_url,
            "--source", str(source), "--out-root", str(tmp_path / "data"), "--num-samples", "2"]  # fmt: skip
    out = generate_api.main(args)
    full = out.read_text().splitlines()
    # Simulate a crash: first row complete, second row cut off half-way.
    out.write_text(full[0] + "\n" + full[1][:20])
    n_before = len(server.requests)
    generate_api.main([*args, "--resume"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == [0, 1, 2]
    assert len(server.requests) - n_before == 2 * 2  # only questions 1 and 2 regenerated (greedy + samples each)


@pytest.mark.usefixtures("api_cache")  # writes the cache file this test scores
def test_score_api_cache_saves_results(tmp_path: Path) -> None:
    cache = tmp_path / "data" / "mock-model" / "triviaqa_3.jsonl"
    results = tmp_path / "results"
    summary = score_api_cache.main([
        "--cache", str(cache), "--results-root", str(results),
        "--methods", "verbalized_2s_api", "p_true_empirical_api",
    ])  # fmt: skip
    for method in ("verbalized_2s_api", "p_true_empirical_api"):
        record = json.loads((results / "mock-model" / "triviaqa_3" / f"{method}.json").read_text())
        assert record["method"] == method
        assert summary[method]["auroc"] == 1.0  # the wrong answer is the most uncertain
        assert summary[method]["local_auroc"] is None
    assert (results / "mock-model" / "triviaqa_3" / "summary_api.json").exists()

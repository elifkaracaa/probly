"""C1 equivalence: the library's batched self-evaluation gives the old plugin scores.

Compares the new plugins (probly ``HFSelfEvaluator``, batched with left padding) with the week-1 code in
``common/reflexive.py`` (one prompt at a time) on the 5-question cache of Qwen2.5-0.5B. The model must be in
the local Hugging Face cache; otherwise the test is skipped. Set ALLOW_MODEL_DOWNLOAD=1 to download it once.
"""

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch

from common import reflexive
from common.data import load_data
from methods import p_true_brainstorm, p_true_zeroshot

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ATOL = 1e-4  # float32 on CPU: batching changes the sums by about 1e-6


@pytest.fixture(scope="module")
def model_and_tok():
    from probly.representer.sampler.huggingface import load_model

    local_only = not os.environ.get("ALLOW_MODEL_DOWNLOAD")
    try:
        return load_model(
            MODEL_NAME,
            model_kwargs={"dtype": torch.float32, "local_files_only": local_only},
            tokenizer_kwargs={"local_files_only": local_only},
        )
    except OSError:
        pytest.skip(f"{MODEL_NAME} is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def rows():
    return load_data("qwen0.5b", 5)


def test_zeroshot_matches_old_code(model_and_tok, rows):
    model, tok = model_and_tok
    new = p_true_zeroshot.score(rows, model, tok)
    for i, row in enumerate(rows):
        old = reflexive.p_true_details(model, tok, row["question"], row["greedy_answer"], mode="plain_ab")
        assert new["uncertainty"][i] == pytest.approx(1.0 - old["p_true"], abs=ATOL)
        assert new["coverage"][i] == pytest.approx(old["coverage"], abs=ATOL)
    assert not any(new["failed"])
    assert new["ids"] == [row["id"] for row in rows]


def test_brainstorm_matches_old_code(model_and_tok, rows):
    model, tok = model_and_tok
    new = p_true_brainstorm.score(rows, model, tok)
    for i, row in enumerate(rows):
        old = reflexive.p_true_details(
            model, tok, row["question"], row["greedy_answer"], brainstorm=row["samples"], mode="plain_ab"
        )
        assert new["uncertainty"][i] == pytest.approx(1.0 - old["p_true"], abs=ATOL)
        assert new["coverage"][i] == pytest.approx(old["coverage"], abs=ATOL)
    assert not any(new["failed"])


def test_brainstorm_without_samples_is_failed(model_and_tok, rows):
    model, tok = model_and_tok
    broken = [dict(row) for row in rows]
    broken[0]["samples"] = []
    new = p_true_brainstorm.score(broken, model, tok)
    assert new["failed"][0] is True
    assert new["uncertainty"][0] == 1.0
    assert not any(new["failed"][1:])

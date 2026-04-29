from __future__ import annotations

from unittest.mock import patch, MagicMock
import json

from build_tfidf.hyde import generate_hypothetical


def test_hyde_ollama():
    resp_data = {"response": "A hypothetical answer about embeddings."}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(resp_data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = generate_hypothetical("What are embeddings?", "ollama")

    assert result == "A hypothetical answer about embeddings."


def test_hyde_fastembed_returns_raw_query():
    result = generate_hypothetical("What are embeddings?", "fastembed")
    assert result == "What are embeddings?"


def test_hyde_openai():
    mock_choice = MagicMock()
    mock_choice.message.content = "A hypothetical answer from OpenAI."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        result = generate_hypothetical("What are embeddings?", "openai")

    assert result == "A hypothetical answer from OpenAI."

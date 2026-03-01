"""Tests for domain config loading and LLM abstraction."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from kg.config import DomainConfig, load_config, CONFIGS_DIR, _build_config
from kg.llm import OpenAIAdapter, AnthropicAdapter, with_retry


# ---------------------------------------------------------------------------
# DomainConfig basics
# ---------------------------------------------------------------------------

class TestDomainConfig:
    def test_default_values(self):
        cfg = DomainConfig()
        assert cfg.name == "math"
        assert cfg.collection_names == ["lit_review"]
        assert cfg.ingest_collection == "lit_review"

    def test_build_from_dict(self):
        raw = {
            "name": "test-domain",
            "extraction_prompt": "You are a test extractor.",
            "normalize": {"foo": "bar"},
            "type_colors": {"algo": "#FFF"},
        }
        cfg = _build_config(raw)
        assert cfg.name == "test-domain"
        assert cfg.extraction_prompt == "You are a test extractor."
        assert cfg.normalize == {"foo": "bar"}
        assert cfg.type_colors == {"algo": "#FFF"}

    def test_ignores_unknown_keys(self):
        raw = {"name": "test", "unknown_field": "value"}
        cfg = _build_config(raw)
        assert cfg.name == "test"
        assert not hasattr(cfg, "unknown_field")


# ---------------------------------------------------------------------------
# Config loading — three-tier resolution
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_explicit_config_path(self, tmp_path):
        config = {
            "name": "custom",
            "extraction_prompt": "Custom prompt",
            "normalize": {"a": "b"},
            "type_colors": {"x": "#000"},
        }
        config_file = tmp_path / "custom.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        cfg = load_config(config_path=str(config_file))
        assert cfg.name == "custom"
        assert cfg.extraction_prompt == "Custom prompt"

    def test_explicit_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/path.yaml")

    def test_domain_yaml_pointer(self, tmp_path):
        # Create a domain.yaml that points to "math"
        domain_yaml = tmp_path / "domain.yaml"
        with open(domain_yaml, 'w') as f:
            yaml.dump({"config": "math"}, f)

        cfg = load_config(data_dir=str(tmp_path))
        assert cfg.name == "math"
        assert len(cfg.normalize) > 0

    def test_domain_yaml_with_overrides(self, tmp_path):
        domain_yaml = tmp_path / "domain.yaml"
        with open(domain_yaml, 'w') as f:
            yaml.dump({
                "config": "math",
                "name": "math-custom",
            }, f)

        cfg = load_config(data_dir=str(tmp_path))
        assert cfg.name == "math-custom"
        # Should still have math's normalize table
        assert "bailey's lemma" in cfg.normalize

    def test_fallback_to_math(self):
        cfg = load_config()
        assert cfg.name == "math"
        assert len(cfg.extraction_prompt) > 0

    def test_explicit_overrides_domain_yaml(self, tmp_path):
        # domain.yaml points to math
        domain_yaml = tmp_path / "domain.yaml"
        with open(domain_yaml, 'w') as f:
            yaml.dump({"config": "math"}, f)

        # Explicit config points to evo
        evo_path = CONFIGS_DIR / "evolutionary-computation.yaml"
        cfg = load_config(config_path=str(evo_path), data_dir=str(tmp_path))
        assert cfg.name == "evolutionary-computation"


class TestBuiltinConfigs:
    """Verify all three built-in configs load correctly."""

    def test_math_config(self):
        cfg = load_config(config_path=str(CONFIGS_DIR / "math.yaml"))
        assert cfg.name == "math"
        assert "mathematical knowledge extractor" in cfg.extraction_prompt
        assert "bailey's lemma" in cfg.normalize
        assert "theorem" in cfg.type_colors

    def test_evolutionary_computation_config(self):
        cfg = load_config(config_path=str(CONFIGS_DIR / "evolutionary-computation.yaml"))
        assert cfg.name == "evolutionary-computation"
        assert "evolutionary computation" in cfg.extraction_prompt
        assert "map elites" in cfg.normalize
        assert cfg.normalize["map elites"] == "MAP-Elites"
        assert "algorithm" in cfg.type_colors

    def test_knowledge_graphs_config(self):
        cfg = load_config(config_path=str(CONFIGS_DIR / "knowledge-graphs.yaml"))
        assert cfg.name == "knowledge-graphs"
        assert "computer science" in cfg.extraction_prompt
        assert "knowledge graph" in cfg.normalize
        assert "method" in cfg.type_colors


# ---------------------------------------------------------------------------
# LLM Adapters
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    def test_chat_calls_openai(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter(client=mock_client)
        result = adapter.chat("system", "prompt")
        assert result == "test response"
        mock_client.chat.completions.create.assert_called_once()

    def test_json_mode_sets_response_format(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = '{"key": "value"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter(client=mock_client)
        adapter.chat("system", "prompt", json_mode=True)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestAnthropicAdapter:
    def test_chat_calls_anthropic(self):
        mock_client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "claude response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        adapter = AnthropicAdapter(client=mock_client)
        result = adapter.chat("system", "prompt")
        assert result == "claude response"
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestWithRetry:
    def test_succeeds_first_try(self):
        result = with_retry(lambda: 42)
        assert result == 42

    def test_retries_on_rate_limit(self):
        call_count = 0
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 rate limit exceeded")
            return "ok"
        result = with_retry(flaky, max_retries=3)
        assert result == "ok"

    def test_raises_non_retryable_error(self):
        def always_fail():
            raise ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            with_retry(always_fail, max_retries=3)

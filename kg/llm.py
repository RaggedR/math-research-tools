"""Minimal LLM abstraction for OpenAI and Anthropic APIs."""

import json
import logging
import random
import time

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """Thin wrapper around the OpenAI chat API."""

    def __init__(self, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client

    def chat(self, system, prompt, model="gpt-4o-mini", max_tokens=2000,
             temperature=0.2, json_mode=False):
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class AnthropicAdapter:
    """Thin wrapper around the Anthropic messages API."""

    def __init__(self, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def chat(self, system, prompt, model="claude-sonnet-4-5-20250929",
             max_tokens=4096, temperature=0.3, json_mode=False):
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text


def with_retry(fn, max_retries=3):
    """Retry with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ('rate', '429', 'limit', 'overloaded')):
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Rate limited, waiting %.1fs...", wait)
                time.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(1)
    return fn()  # final attempt

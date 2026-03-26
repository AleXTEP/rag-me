"""
HyDE (Hypothetical Document Embeddings) implementation.

Instead of embedding the query directly, generates a hypothetical answer
using an LLM and embeds that instead. The hypothetical answer lives closer
in embedding space to actual relevant chunks than a short query does.
"""

from config import HYDE_PROVIDER, HYDE_MODEL, HYDE_API_KEY, HYDE_LOCAL_URL
import logging

logger = logging.getLogger(__name__)

_PROMPT = (
    "Write a concise passage (3-5 sentences) that directly answers "
    "the following question. "
    "Do not include the question itself.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


def generate_hypothetical_document(query: str) -> str:
    """Generate a hypothetical answer for the query using the configured LLM provider."""
    prompt = _PROMPT.format(query=query)
    provider = HYDE_PROVIDER.lower()
    if provider == "claude":
        return _claude(prompt)
    elif provider == "openai":
        return _openai(prompt)
    elif provider == "local":
        return _local(prompt)
    else:
        raise ValueError(f"Unknown HYDE_PROVIDER: {provider!r}. Use 'claude', 'openai', or 'local'.")


def _claude(prompt: str) -> str:
    import anthropic
    key = HYDE_API_KEY or None
    client = anthropic.Anthropic(api_key=key)
    logger.info(f"Generating hypothetical document for query: {prompt}")
    message = client.messages.create(
        model=HYDE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()


def _openai(prompt: str) -> str:
    from openai import OpenAI
    key = HYDE_API_KEY or None
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def _local(prompt: str) -> str:
    """Call a local Ollama instance."""
    import httpx
    response = httpx.post(
        f"{HYDE_LOCAL_URL}/api/generate",
        json={"model": HYDE_MODEL, "prompt": prompt, "stream": False},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

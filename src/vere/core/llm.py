import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from vere.config import load_registry

def available_models() -> list[str]:
    return list(load_registry().models)

def get_llm(name: str | None = None, temperature: float = 0.0, **kwargs):
    reg = load_registry()
    spec = reg.models.get(name or reg.active)
    if spec is None:
        raise KeyError(f"unknown model '{name}', available: {available_models()}")
    if spec.provider == "ollama":
        return ChatOllama(
            model=spec.model,
            temperature=temperature,
            **(spec.options or {}),
            **kwargs,
        )
    api_key = os.environ.get(spec.api_key_env or "OPENAI_API_KEY", "not-set")
    return ChatOpenAI(
            model=spec.model,
            base_url=spec.base_url,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )

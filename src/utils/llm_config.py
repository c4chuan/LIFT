from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer, mostly
            for Hugging Face transformers.
        mode: The mode of the API calls, e.g., "chat" or "generation".
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    mode: str | None = None
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)


def construct_llm_config(provider = "openai") -> LMConfig:
    llm_config = LMConfig(
        provider=provider, model="gpt-4o", mode="chat"
    )
    if provider in ["openai", "google", "sglang", "azure"]:
        llm_config.gen_config["temperature"] = 0.0
        llm_config.gen_config["top_p"] = 1.0
        llm_config.gen_config["context_length"] = 0
        llm_config.gen_config["max_tokens"] = 4096
        llm_config.gen_config["stop_token"] = None
        llm_config.gen_config["max_obs_length"] = 3840
        llm_config.gen_config["max_retry"] = 1
    else:
        raise NotImplementedError(f"provider {provider} not implemented")
    return llm_config
from __future__ import annotations

from collections.abc import Generator

import anthropic

from rpg_bot.config import get_settings
from rpg_bot.llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_RAG


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm.model
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature
        self.history: list[dict[str, str]] = []

    def _get_system_prompt(self, context: str | None = None) -> str:
        if context:
            return SYSTEM_PROMPT_WITH_RAG.format(context=context)
        return SYSTEM_PROMPT

    def chat_stream(
        self,
        user_message: str,
        context: str | None = None,
    ) -> Generator[str, None, None]:
        self.history.append({"role": "user", "content": user_message})

        full_response = ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self._get_system_prompt(context),
            messages=self.history,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield text

        self.history.append({"role": "assistant", "content": full_response})

    def chat(
        self,
        user_message: str,
        context: str | None = None,
    ) -> str:
        return "".join(self.chat_stream(user_message, context))

    def clear_history(self) -> None:
        self.history.clear()

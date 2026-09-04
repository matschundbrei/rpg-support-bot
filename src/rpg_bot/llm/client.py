from __future__ import annotations

from collections.abc import Generator
from typing import cast

from anthropic import Anthropic
from anthropic.types import MessageParam
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from rpg_bot.config import get_settings
from rpg_bot.llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_RAG


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.backend = settings.llm.backend
        self.model = settings.llm.model
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature
        self.max_history = settings.llm.max_history
        self.history: list[dict[str, str]] = []

        self.client: Anthropic | OpenAI
        if self.backend == "anthropic":
            self.client = Anthropic(api_key=settings.anthropic_api_key)
        elif self.backend == "openai":
            self.client = OpenAI(
                api_key=settings.openai_api_key or "no-key-required",
                base_url=settings.llm.base_url or None,
            )
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend!r}")

    def _get_system_prompt(self, context: str | None = None) -> str:
        if context:
            return SYSTEM_PROMPT_WITH_RAG.format(context=context)
        return SYSTEM_PROMPT

    def _stream_anthropic(
        self, client: Anthropic, system: str, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        with client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=cast("list[MessageParam]", messages),
            extra_body={"temperature": self.temperature},
        ) as stream:
            yield from stream.text_stream

    def _stream_openai(
        self, client: OpenAI, system: str, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        oai_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            *cast("list[ChatCompletionMessageParam]", messages),
        ]
        stream = client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def _trim_history(self) -> None:
        """Keep only the most recent max_history messages, preserving pairs."""
        if self.max_history > 0 and len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
            # The first message sent to the API must be a user message
            if self.history and self.history[0]["role"] != "user":
                self.history.pop(0)

    def chat_stream(
        self,
        user_message: str,
        context: str | None = None,
    ) -> Generator[str, None, None]:
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()
        system = self._get_system_prompt(context)

        full_response = ""
        try:
            client = self.client
            if isinstance(client, Anthropic):
                streamer = self._stream_anthropic(client, system, self.history)
            else:
                streamer = self._stream_openai(client, system, self.history)

            for text in streamer:
                full_response += text
                yield text
        except Exception:
            # Roll back the user message so a failed call doesn't leave a
            # dangling user turn (the next call would send two consecutive
            # user messages, which the APIs reject).
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            raise

        self.history.append({"role": "assistant", "content": full_response})

    def chat(
        self,
        user_message: str,
        context: str | None = None,
    ) -> str:
        return "".join(self.chat_stream(user_message, context))

    def clear_history(self) -> None:
        self.history.clear()

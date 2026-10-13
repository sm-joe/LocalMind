"""Thin Ollama adapter with bounded generation for predictable local latency."""

from ollama import Client


class LLMService:
    def __init__(self, base_url: str, model: str) -> None:
        self.client = Client(host=base_url)
        self.model = model

    def generate_stream(self, prompt: str):
        """Yield text deltas from Ollama as they are generated."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        stream = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "temperature": 0.1,
                "num_predict": 320,
            },
        )
        emitted = False
        for part in stream:
            message = getattr(part, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content:
                emitted = True
                yield content
        if not emitted:
            raise RuntimeError("Ollama returned an empty streaming response.")

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                # Keep local generation bounded. Citation formatting is now
                # assigned by the backend and no repair-generation call is needed.
                "num_predict": 320,
            },
        )
        content = response.message.content
        if not content or not content.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return content.strip()

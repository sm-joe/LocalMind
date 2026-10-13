"""Thin Ollama adapter with bounded generation for predictable local latency."""

from ollama import Client


class LLMService:
    def __init__(self, base_url: str, model: str) -> None:
        self.client = Client(host=base_url)
        self.model = model

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

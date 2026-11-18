from ollama import Client


class LLMService:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int = 6144,
    ) -> None:
        if num_ctx < 2048:
            raise ValueError("Ollama context must be at least 2048 tokens.")
        self.client = Client(host=base_url)
        self.model = model
        self.num_ctx = num_ctx

    def generate(
        self,
        prompt: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = self.client.chat(
            model=self.model,
            # Retrieval prompts contain multiple source passages and citation
            # markers. Ollama's default context may truncate those passages.
            options={
                "num_ctx": self.num_ctx,
                "num_predict": 512,
                "temperature": 0.2,
            },
            # Qwen3 is a reasoning-capable model. Its default thinking mode
            # can spend most of a CPU-only request on hidden deliberation
            # rather than emitting a cited answer. Use direct generation for
            # interactive document Q&A. Older non-thinking models omit this.
            **({"think": False} if self.model.lower().startswith("qwen3") else {}),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.message.content

        if not content or not content.strip():
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return content.strip()
    def stream(self, prompt: str):
        """Stream Ollama tokens as they are generated (reference project flow)."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        # Do not collect a full response before yielding output: CPU inference
        # must show progress in the browser as soon as Ollama emits text.
        options = {
            "num_ctx": self.num_ctx,
            "num_predict": 400,
            "temperature": 0.2,
        }
        for item in self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options=options,
            **({"think": False} if self.model.lower().startswith("qwen3") else {}),
        ):
            token = item.get("response", "") if isinstance(item, dict) else getattr(item, "response", "")
            if token:
                yield token

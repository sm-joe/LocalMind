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
            options={"num_ctx": self.num_ctx},
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
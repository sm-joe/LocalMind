
from ollama import Client


class EmbeddingService:
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        base_url: str,
        model: str,
    ) -> None:
        self.client = Client(host=base_url)
        self.model = model

    def embed(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Cannot generate an embedding for empty text."
            )

        response = self.client.embed(
            model=self.model,
            input=text,
        )

        embeddings = response.embeddings

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        return embeddings[0]

    def embed_many(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Cannot generate an embedding for empty text."
            )

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            response = self.client.embed(
                model=self.model,
                input=batch,
            )

            embeddings = response.embeddings

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Ollama returned an unexpected number "
                    "of embeddings."
                )

            all_embeddings.extend(embeddings)

        return all_embeddings

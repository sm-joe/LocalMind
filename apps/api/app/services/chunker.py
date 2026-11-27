import re
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    normalized_text = text.strip()

    if not normalized_text:
        return []

    paragraphs = _extract_paragraphs(
        normalized_text
    )

    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current: list[tuple[str, int, int]] = []

    for paragraph, start, end in paragraphs:
        for part, part_start, part_end in _split_paragraph(
            paragraph, start, end, chunk_size
        ):
            joined_length = sum(len(item[0]) + 2 for item in current) + len(part)
            if joined_length > chunk_size:
                chunks.append(_build_chunk(len(chunks), current))
                # Keep whole trailing parts only when they fit beside new text.
                overlap: list[tuple[str, int, int]] = []
                available = min(chunk_overlap, chunk_size - len(part) - 2)
                for item in reversed(current):
                    if len(item[0]) > available:
                        break
                    overlap.append(item)
                    available -= len(item[0]) + 2
                current = list(reversed(overlap))
            current.append((part, part_start, part_end))

    if current:
        chunks.append(_build_chunk(len(chunks), current))
    return chunks


def _split_paragraph(
    paragraph: str, start: int, end: int, chunk_size: int,
) -> Iterator[tuple[str, int, int]]:
    if len(paragraph) <= chunk_size:
        yield paragraph, start, end
        return

    cursor = 0
    for part in _split_long_paragraph(paragraph, chunk_size):
        # The splitter can normalize whitespace. Locate each piece from the
        # previous match so repeated phrases point to their actual occurrence.
        pattern = re.compile(r"\s+".join(re.escape(word) for word in part.split()))
        match = pattern.search(paragraph, cursor)
        if match is None:
            raise RuntimeError("Could not locate split text in its source paragraph.")
        yield part, start + match.start(), start + match.end()
        cursor = match.end()


def _extract_paragraphs(
    text: str,
) -> list[tuple[str, int, int]]:
    paragraphs: list[
        tuple[str, int, int]
    ] = []

    paragraph_pattern = re.compile(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
        re.DOTALL,
    )

    for match in paragraph_pattern.finditer(text):
        paragraph = match.group(0).strip()

        if not paragraph:
            continue

        leading_whitespace = (
            len(match.group(0))
            - len(match.group(0).lstrip())
        )

        start = (
            match.start()
            + leading_whitespace
        )

        end = start + len(paragraph)

        paragraphs.append(
            (
                paragraph,
                start,
                end,
            )
        )

    return paragraphs


def _split_long_paragraph(
    paragraph: str,
    chunk_size: int,
) -> list[str]:
    sentences = _split_sentences(
        paragraph
    )

    if len(sentences) == 1:
        return _split_words(
            paragraph,
            chunk_size,
        )

    parts: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current:
                parts.append(current)
                current = ""

            parts.extend(
                _split_words(
                    sentence,
                    chunk_size,
                )
            )
            continue

        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            parts.append(current)
            current = sentence

    if current:
        parts.append(current)

    return parts


def _split_sentences(
    text: str,
) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]


def _split_words(
    text: str,
    chunk_size: int,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    parts: list[str] = []
    current = ""

    for word in words:
        if len(word) > chunk_size:
            if current:
                parts.append(current)
                current = ""

            start = 0

            while start < len(word):
                parts.append(
                    word[
                        start : start + chunk_size
                    ]
                )
                start += chunk_size

            continue

        if not current:
            current = word
            continue

        candidate = f"{current} {word}"

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            parts.append(current)
            current = word

    if current:
        parts.append(current)

    return parts


def _build_chunk(
    chunk_id: int,
    parts: list[tuple[str, int, int]],
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text="\n\n".join(part for part, _, _ in parts),
        start_char=parts[0][1],
        end_char=parts[-1][2],
    )

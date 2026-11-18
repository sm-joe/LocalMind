from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedDocument:
    filename: str
    content_type: str
    text: str


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

MAX_PDF_PAGES = 200
MAX_EXTRACTED_TEXT_CHARS = 2_000_000
MAX_DOCX_PARAGRAPHS = 10_000


def parse_document(
    file_path: str | Path,
    content_type: str,
) -> ParsedDocument:
    path = Path(file_path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    if not path.is_file():
        raise ValueError(
            "Document file does not exist."
        )

    extension = path.suffix.lower()

    if extension == ".pdf":
        text = _parse_pdf(path)
    elif extension == ".docx":
        text = _parse_docx(path)
    elif extension in {".txt", ".md"}:
        text = _parse_text(path)
    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    normalized_text = _normalize_text(
        text, preserve_paragraphs=(extension == ".docx")
    )

    if not normalized_text:
        raise ValueError(
            "The document contains no extractable text."
        )

    if len(normalized_text) > MAX_EXTRACTED_TEXT_CHARS:
        raise ValueError(
            "The document contains too much extracted text."
        )

    return ParsedDocument(
        filename=path.name,
        content_type=content_type,
        text=normalized_text,
    )


def _parse_pdf(path: Path) -> str:
    import fitz

    pages: list[str] = []

    try:
        with fitz.open(path) as document:
            page_count = len(document)

            if page_count > MAX_PDF_PAGES:
                raise ValueError(
                    "PDF contains too many pages."
                )

            for page in document:
                pages.append(page.get_text())

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            "The PDF document could not be parsed."
        ) from exc

    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(path)

        if len(document.paragraphs) > MAX_DOCX_PARAGRAPHS:
            raise ValueError(
                "DOCX document contains too many paragraphs."
            )

        # document.paragraphs silently omits all content inside Word tables.
        # Read both paragraphs and tables in their true body order, including
        # tables containing interview questions and their answers.
        sections: list[str] = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    sections.append(block.text)
            elif isinstance(block, Table):
                for row in block.rows:
                    # Preserve paired question/answer columns and avoid
                    # emitting duplicate text for horizontally merged cells.
                    cells: list[str] = []
                    seen_cells: set[int] = set()
                    for cell in row.cells:
                        cell_identity = id(cell._tc)
                        if cell_identity in seen_cells:
                            continue
                        seen_cells.add(cell_identity)
                        text = " ".join(
                            paragraph.text.strip()
                            for paragraph in cell.paragraphs
                            if paragraph.text.strip()
                        )
                        if text:
                            cells.append(text)
                    if cells:
                        sections.append(" | ".join(cells))

        return "\n\n".join(sections)

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            "The DOCX document could not be parsed."
        ) from exc

def _parse_text(path: Path) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except OSError as exc:
        raise ValueError(
            "The text document could not be read."
        ) from exc


def _normalize_text(
    text: str, preserve_paragraphs: bool = False
) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
    ]
    if not preserve_paragraphs:
        return "\n".join(line for line in lines if line).strip()

    # The chunker recognizes paragraph boundaries only when they are
    # separated by a blank line. Do not flatten DOCX questions and answers
    # into one giant paragraph before splitting them into retrieval chunks.
    normalized: list[str] = []
    blank = False
    for line in lines:
        if not line:
            blank = bool(normalized)
        else:
            if blank:
                normalized.append("")
                blank = False
            normalized.append(line)
    return "\n".join(normalized).strip()
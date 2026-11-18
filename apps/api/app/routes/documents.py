import tempfile
import zipfile
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.document_registry import DocumentIndexRegistry, DocumentIndexInProgress
from app.services.vector_store import VectorStore


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

logger = logging.getLogger(__name__)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    status: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    status: str


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_FILENAME_LENGTH = 255
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
MAX_DOCX_ARCHIVE_ENTRIES = 1000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


def create_vector_store() -> VectorStore:
    settings = get_settings()

    return VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )


def create_ingestion_service() -> IngestionService:
    settings = get_settings()

    embedding_service = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )

    vector_store = create_vector_store()

    return IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        registry=DocumentIndexRegistry(),
    )


def create_document_registry() -> DocumentIndexRegistry:
    return DocumentIndexRegistry()


def _sanitize_filename(filename: str) -> str:
    normalized_filename = filename.replace("\\", "/")
    sanitized_filename = Path(normalized_filename).name

    if not sanitized_filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if "\x00" in sanitized_filename:
        raise HTTPException(
            status_code=400,
            detail="Filename contains invalid characters.",
        )

    if len(sanitized_filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Filename is too long.",
        )

    return sanitized_filename


async def _write_upload_to_temporary_file(
    file: UploadFile,
    temporary_file,
) -> int:
    total_size = 0

    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "File is too large. "
                    "Maximum upload size is 10 MiB."
                ),
            )

        temporary_file.write(chunk)

    return total_size


def _validate_pdf(path: Path) -> None:
    try:
        with path.open("rb") as file:
            header = file.read(len(PDF_MAGIC))

    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF could not be read.",
        ) from exc

    if header != PDF_MAGIC:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF.",
        )


def _validate_docx(path: Path) -> None:
    try:
        with path.open("rb") as file:
            header = file.read(len(ZIP_MAGIC))

        if header != ZIP_MAGIC:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid DOCX.",
            )

        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()

            if len(entries) > MAX_DOCX_ARCHIVE_ENTRIES:
                raise HTTPException(
                    status_code=400,
                    detail="DOCX archive contains too many entries.",
                )

            total_uncompressed_bytes = sum(
                entry.file_size for entry in entries
            )

            if (
                total_uncompressed_bytes
                > MAX_DOCX_UNCOMPRESSED_BYTES
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "DOCX archive expands beyond the "
                        "allowed size."
                    ),
                )

            names = {entry.filename for entry in entries}

            if (
                "[Content_Types].xml" not in names
                or "word/document.xml" not in names
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a valid DOCX.",
                )

            if archive.testzip() is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded DOCX archive is corrupted.",
                )

    except HTTPException:
        raise
    except (
        OSError,
        zipfile.BadZipFile,
        RuntimeError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid DOCX.",
        ) from exc


def _validate_text(path: Path) -> None:
    try:
        path.read_text(
            encoding="utf-8",
        )

    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded text file is not valid UTF-8.",
        ) from exc

    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded text file could not be read.",
        ) from exc


def _validate_file_content(
    path: Path,
    extension: str,
) -> None:
    if extension == ".pdf":
        _validate_pdf(path)
        return

    if extension == ".docx":
        _validate_docx(path)
        return

    if extension in {".txt", ".md"}:
        _validate_text(path)
        return

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file type: {extension}"
        ),
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    reindex: bool = Query(default=False),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    original_filename = _sanitize_filename(
        file.filename
    )

    extension = Path(
        original_filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}"
            ),
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            total_size = await _write_upload_to_temporary_file(
                file=file,
                temporary_file=temporary_file,
            )

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        _validate_file_content(
            path=temporary_path,
            extension=extension,
        )

        service = create_ingestion_service()

        result = service.ingest(
            file_path=temporary_path,
            content_type=file.content_type
            or "application/octet-stream",
            original_filename=original_filename,
            **({"force_reindex": True} if reindex is True else {}),
        )

        status = (
            "already_indexed"
            if result.already_indexed
            else "indexed"
        )

        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunks=result.chunks,
            status=status,
        )

    except HTTPException:
        raise

    except DocumentIndexInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(
            "Document ingestion failed; exception_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )

        await file.close()



@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents() -> DocumentListResponse:
    try:
        vector_store = create_vector_store()
        settings = get_settings()

        collections = vector_store.client.get_collections()

        collection_exists = any(
            collection.name == settings.qdrant_collection
            for collection in collections.collections
        )

        if not collection_exists:
            return DocumentListResponse(documents=[])

        document_metadata: dict[str, dict[str, str | int]] = {}

        for point in vector_store.iter_points(limit=1000):
            payload = point.payload or {}
            raw_document_id = payload.get("document_id")

            if not raw_document_id:
                continue

            document_id = str(raw_document_id)

            if document_id not in document_metadata:
                document_metadata[document_id] = {
                    "filename": str(payload.get("filename") or "unknown"),
                    "content_type": str(
                        payload.get("content_type")
                        or "application/octet-stream"
                    ),
                    "chunks": 0,
                }

            document_metadata[document_id]["chunks"] += 1

        documents = [
            DocumentResponse(
                document_id=document_id,
                filename=str(metadata["filename"]),
                content_type=str(metadata["content_type"]),
                chunks=int(metadata["chunks"]),
            )
            for document_id, metadata in document_metadata.items()
        ]

        return DocumentListResponse(documents=documents)

    except Exception as exc:
        logger.error(
            "Document listing failed; exception_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents.",
        ) from exc



class DocumentIndexStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks: int


@router.get(
    "/status/{document_id}",
    response_model=DocumentIndexStatusResponse,
)
def document_index_status(document_id: str) -> DocumentIndexStatusResponse:
    record = create_document_registry().get_status(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document indexing state not found.")
    return DocumentIndexStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,
        chunks=record.chunks,
    )


class DocumentChunkResponse(BaseModel):
    document_id: str
    filename: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int


@router.get(
    "/{document_id}/chunks/{chunk_id}",
    response_model=DocumentChunkResponse,
)
def get_document_chunk(
    document_id: str,
    chunk_id: int,
) -> DocumentChunkResponse:
    if not document_id.strip() or chunk_id < 0:
        raise HTTPException(status_code=400, detail="Invalid chunk identifier.")
    try:
        payload = create_vector_store().get_document_chunk(document_id, chunk_id)
    except Exception as exc:
        logger.exception("Failed to fetch document chunk.")
        raise HTTPException(status_code=503, detail="Document store unavailable.") from exc
    if not payload or str(payload.get("document_id")) != document_id or payload.get("chunk_id") != chunk_id:
        raise HTTPException(status_code=404, detail="Source passage not found.")
    return DocumentChunkResponse(
        document_id=document_id,
        filename=str(payload.get("filename", "unknown")),
        chunk_id=chunk_id,
        text=str(payload.get("text", "")),
        start_char=int(payload.get("start_char", 0)),
        end_char=int(payload.get("end_char", 0)),
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
def delete_document(
    document_id: str,
) -> DocumentDeleteResponse:
    if not document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Document ID is required.",
        )

    try:
        vector_store = create_vector_store()

        vector_store.delete_document(
            document_id
        )
        create_document_registry().forget(document_id)

        return DocumentDeleteResponse(
            document_id=document_id,
            status="deleted",
        )

    except Exception as exc:
        logger.error(
            "Document deletion failed; exception_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document.",
        ) from exc
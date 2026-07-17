import argparse
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_embedding_model(settings: Settings):
    if settings.google_api_key:
        return GoogleGenerativeAIEmbeddings(
            api_key=settings.google_api_key,
            model=settings.google_embedding_model,
        )

    if settings.openai_api_key:
        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    raise RuntimeError(
        "No embedding provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY."
    )


def load_pdf(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info("Loading document: %s", path)
    return PyPDFLoader(str(path)).load()


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Generated %d document chunks.", len(chunks))
    return chunks


def clean_documents(documents: list[Document]) -> list[Document]:
    cleaned = []
    for document in documents:
        metadata = {
            key: value
            for key, value in document.metadata.items()
            if value not in ("", None)
        }
        cleaned.append(Document(page_content=document.page_content, metadata=metadata))
    return cleaned


def create_vector_store(settings: Settings, overwrite: bool = False) -> PGVector:
    embeddings = build_embedding_model(settings)
    return PGVector(
        embeddings=embeddings,
        collection_name=settings.pgvector_collection,
        connection=settings.pgvector_url,
        use_jsonb=True,
        pre_delete_collection=overwrite,
    )


def ingest_pdf(settings: Settings, overwrite: bool = False) -> None:
    documents = load_pdf(settings.pdf_path)
    if not documents:
        logger.warning("No pages found in %s.", settings.pdf_path)
        return

    chunks = split_documents(documents)
    if not chunks:
        logger.warning("No document chunks were generated.")
        return

    enriched = clean_documents(chunks)
    ids = [f"doc-{index}" for index in range(len(enriched))]

    store = create_vector_store(settings, overwrite=overwrite)
    store.add_documents(documents=enriched, ids=ids)

    logger.info(
        "Ingested %d documents into collection '%s'.",
        len(enriched),
        settings.pgvector_collection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF and store embeddings in PostgreSQL + pgvector.")
    parser.add_argument(
        "--pdf-path",
        help="Path to the PDF file to ingest.",
    )
    parser.add_argument(
        "--collection",
        help="PGVector collection name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing PGVector collection before ingesting.",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.pdf_path:
        settings.pdf_path = Path(args.pdf_path).expanduser().resolve()
    if args.collection:
        settings.pgvector_collection = args.collection

    ingest_pdf(settings, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

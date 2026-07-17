import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


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


def build_retriever(settings: Settings):
    embeddings = build_embedding_model(settings)
    store = PGVector(
        embeddings=embeddings,
        collection_name=settings.pgvector_collection,
        connection=settings.pgvector_url,
        use_jsonb=True,
    )

    retriever = store.as_retriever(search_kwargs={"k": 10})

    retriever._store = store
    retriever._top_k = 10

    return retriever


def build_llm(settings: Settings):
    provider = settings.llm_provider()
    if provider == "google":
        return GoogleGenerativeAI(
            api_key=settings.google_api_key,
            model=settings.google_llm_model,
            temperature=0.0,
        )

    if provider == "openai":
        return OpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_llm_model,
            temperature=0.0,
        )

    raise RuntimeError(
        "No LLM provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY."
    )


def search_with_scores(query: str, k: int = 10) -> list:
    """Search using similarity_search_with_score to get relevance scores."""
    settings = Settings()
    embeddings = build_embedding_model(settings)
    store = PGVector(
        embeddings=embeddings,
        collection_name=settings.pgvector_collection,
        connection=settings.pgvector_url,
        use_jsonb=True,
    )
    
    results = store.similarity_search_with_score(query, k=k)
    for doc, score in results:
        logger.info(f"[Score: {score:.4f}] {doc.metadata.get('source', 'N/A')}")
    
    return results


def search_prompt(question=None):
    settings = Settings()
    retriever = build_retriever(settings)
    llm = build_llm(settings)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    if question:
        result = chain.invoke({"query": question})
        return result.get("result", result)
    return chain

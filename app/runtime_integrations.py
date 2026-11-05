from app.config import Settings
from app.embeddings import EmbeddingProvider, HashEmbeddingProvider
from app.external_embeddings import OpenAICompatibleEmbeddingProvider
from app.milvus_client import PyMilvusClientAdapter
from app.reranking import LexicalReranker, Reranker
from app.siliconflow_reranker import SiliconFlowReranker
from app.vector_store import MilvusVectorStore, VectorStore


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(dimension=settings.embedding_dimension)
    if not settings.embed_api_key:
        raise ValueError("EMBED_API_KEY required for openai-compatible embeddings")
    base_url = settings.embedding_base_url or settings.chat_base_url
    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL or CHAT_BASE_URL is required")
    return OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        model=settings.embedding_model,
        api_key=settings.embed_api_key,
        dimension=settings.embedding_dimension,
    )


def create_reranker(settings: Settings) -> Reranker:
    if settings.rerank_provider == "lexical":
        return LexicalReranker()
    if not settings.rerank_api_key:
        raise ValueError("RERANK_API_KEY required for siliconflow reranking")
    return SiliconFlowReranker(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        api_key=settings.rerank_api_key,
    )


def create_vector_store(settings: Settings) -> VectorStore:
    client = PyMilvusClientAdapter(settings.milvus_uri)
    return MilvusVectorStore(
        client=client,
        collection_name=settings.milvus_collection,
        dimension=settings.embedding_dimension,
    )

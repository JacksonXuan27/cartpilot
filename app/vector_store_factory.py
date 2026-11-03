import importlib
import os
from collections.abc import Callable

from app.config import Settings
from app.vector_store import (
    InMemoryVectorStore,
    MilvusClientLike,
    MilvusVectorStore,
    VectorStore,
)


def create_milvus_client(uri: str) -> MilvusClientLike:
    milvus_client_class = _load_milvus_client_class()
    return milvus_client_class(uri=uri)


def _load_milvus_client_class():
    previous_dotenv_flag = os.environ.get("PYTHON_DOTENV_DISABLED")
    os.environ["PYTHON_DOTENV_DISABLED"] = "true"
    try:
        milvus_module = importlib.import_module("pymilvus")
    finally:
        if previous_dotenv_flag is None:
            os.environ.pop("PYTHON_DOTENV_DISABLED", None)
        else:
            os.environ["PYTHON_DOTENV_DISABLED"] = previous_dotenv_flag
    return milvus_module.MilvusClient


def create_vector_store(
    settings: Settings,
    *,
    dimension: int,
    client_factory: Callable[..., MilvusClientLike] | None = None,
) -> VectorStore:
    if settings.milvus_uri is None or not settings.milvus_uri.strip():
        return InMemoryVectorStore(dimension=dimension)
    if client_factory is None:
        client = create_milvus_client(settings.milvus_uri)
    else:
        client = client_factory(uri=settings.milvus_uri)
    return MilvusVectorStore(
        client=client,
        collection_name=settings.milvus_collection,
        dimension=dimension,
    )

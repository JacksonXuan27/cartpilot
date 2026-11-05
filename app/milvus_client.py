from typing import Any


class PyMilvusClientAdapter:
    def __init__(self, uri: str, client: Any | None = None) -> None:
        self._uri = uri
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self._uri)
        return self._client

    def has_collection(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name=collection_name)

    def create_collection(self, **kwargs: Any) -> Any:
        from pymilvus import DataType

        schema = self.client.create_schema(
            auto_id=kwargs["auto_id"],
            enable_dynamic_field=kwargs["enable_dynamic_field"],
        )
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256
        )
        schema.add_field(
            field_name="document_id", datatype=DataType.VARCHAR, max_length=36
        )
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="metadata", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=kwargs["dimension"],
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=kwargs["metric_type"],
        )
        return self.client.create_collection(
            collection_name=kwargs["collection_name"],
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, **kwargs: Any) -> Any:
        return self.client.upsert(**kwargs)

    def search(self, **kwargs: Any) -> Any:
        return self.client.search(**kwargs)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

from __future__ import annotations

import os
import json
import uuid
from typing import (
    TYPE_CHECKING,
    Tuple,
    Optional,
    Iterable,
    List,
)

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import vantage

METADATA_PREFIX = "meta_"


class Vantage(VectorStore):
    """
    `Vantage` vector store.

    To use, you should have the ``vantage-sdk`` python package installed.
    """

    def __init__(
        self,
        client: vantage.VantageClient,
        embedding: Embeddings,
        collection: Optional[vantage.Collection] = None,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        user_provided_embeddings: Optional[bool] = False,
        llm: Optional[str] = None,
        external_key_id: Optional[str] = None,
    ):
        """Initialize the Vantage vector store."""

        try:
            import vantage
        except ImportError:
            raise ImportError(
                "Could not import vantage python package. "
                "Please install it with `pip install vantage-sdk`."
            )

        self._client = client
        self._embedding = embedding
        self._collection = collection or self._get_or_create_vantage_collection(
            client=self._client,
            collection_id=collection_id,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            user_provided_embeddings=user_provided_embeddings,
            llm=llm,
            external_key_id=external_key_id,
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    @staticmethod
    def _get_vantage_client(
        vantage_jwt_token: Optional[str] = None,
        account_id: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> vantage.VantageClient:
        # TODO: docstring + update auth
        try:
            import vantage
        except ImportError:
            raise ImportError(
                "Could not import vantage python package. "
                "Please install it with `pip install vantage-sdk`"
            )

        vantage_jwt_token = vantage_jwt_token or os.environ.get("VANTAGE_JWT_TOKEN")
        account_id = account_id or os.environ.get("VANTAGE_ACCOUNT_ID")
        vantage_api_key = vantage_api_key or os.environ.get("VANTAGE_API_KEY")

        return vantage.VantageClient.using_jwt_token(
            vantage_api_jwt_token=vantage_jwt_token,
            account_id=account_id,
            vantage_api_key=vantage_api_key,
        )

    @staticmethod
    def _get_or_create_vantage_collection(
        client: Optional[vantage.VantageClient] = None,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        user_provided_embeddings: Optional[bool] = False,
        llm: Optional[str] = None,
        external_key_id: Optional[str] = None,
    ) -> vantage.Collection:
        # TODO: docstring
        try:
            collection = client.get_collection(collection_id=collection_id)
        except:
            collection = client.create_collection(
                collection_id=collection_id,
                collection_name=collection_name
                or f"LangChain Collection [{collection_id}]",
                embeddings_dimension=embedding_dimension,
                user_provided_embeddings=user_provided_embeddings,
                llm=llm,
                external_key_id=external_key_id,
            )

        return collection

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        # TODO: docstring

        vantage_documents = []
        vantage_metadata_dicts = [
            [(METADATA_PREFIX + k, v) for k, v in meta_dict.items()]
            for meta_dict in metadatas
        ]

        ids = ids if ids else [str(uuid.uuid4()) for i in range(len(texts))]

        if self._collection.user_provided_embeddings:
            embeddings = embeddings if embeddings else self._embed_texts(texts)
        else:
            embeddings = [None] * len(texts)

        for id, text, metadata_dict, embedding in zip(
            ids, texts, vantage_metadata_dicts, embeddings
        ):
            document = {
                "id": id,
                "text": text,
            }
            document.update(metadata_dict)
            if self._collection.user_provided_embeddings:
                document.update({"embeddings": embedding})
            vantage_documents.append(document)

        vantage_documents_jsonl = "\n".join(map(json.dumps, vantage_documents))

        self._client.upload_documents_from_jsonl(
            collection_id=str(self._collection.id),
            documents=vantage_documents_jsonl,
        )

        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        # TODO: docstring

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(
            texts,
            metadatas,
            ids,
            embeddings,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        accuracy: Optional[float] = 0.3,
        filter: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> List[Document]:
        # TODO: docstring + use k param

        search_response = self._client.semantic_search(
            text=query,
            collection_id=self._collection.id,
            accuracy=accuracy,
            boolean_filter=filter,
            vantage_api_key=vantage_api_key,
        )

        # TODO: Change result.id to text field
        return [Document(page_content=result.id) for result in search_response.results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        accuracy: Optional[float] = 0.3,
        filter: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        # TODO: docstring + use k param

        search_response = self._client.semantic_search(
            text=query,
            collection_id=self._collection.id,
            accuracy=accuracy,
            boolean_filter=filter,
            vantage_api_key=vantage_api_key,
        )

        # TODO: Change result.id to text field
        return [
            zip(Document(page_content=result.id), result.score)
            for result in search_response.results
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        accuracy: Optional[float] = 0.3,
        filter: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> List[Document]:
        # TODO: docstring + use k param

        search_response = self._client.embedding_search(
            embedding=embedding,
            collection_id=self._collection.id,
            accuracy=accuracy,
            boolean_filter=filter,
            vantage_api_key=vantage_api_key,
        )

        # TODO: Change result.id to text field
        return [Document(page_content=result.id) for result in search_response.results]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        accuracy: Optional[float] = 0.3,
        filter: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        # TODO: docstring + use k param

        search_response = self._client.embedding_search(
            embedding=embedding,
            collection_id=self._collection.id,
            accuracy=accuracy,
            boolean_filter=filter,
            vantage_api_key=vantage_api_key,
        )

        # TODO: Change result.id to text field
        return [
            zip(Document(page_content=result.id), result.score)
            for result in search_response.results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        accuracy: Optional[float] = 0.3,
        filter: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            accuracy: TODO
            filter: TODO
            vantage_api_key: TODO
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        query_embedding = self._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            accuracy=accuracy,
            boolean_filter=filter,
            vantage_api_key=vantage_api_key,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        accuracy: Optional[float] = 0.3,
        filter: Optional[str] = None,
        vantage_api_key: Optional[str] = None,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            accuracy: TODO
            filter: TODO
            vantage_api_key: TODO
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        search_response = self._client.embedding_search(
            embedding=embedding,
            collection_id=self._collection.id,
            accuracy=accuracy,
            boolean_filter=filter,
            vantage_api_key=vantage_api_key,
        )

        mmr_selected_ids = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [item["values"] for item in search_response["matches"]],
            k=fetch_k,
            lambda_mult=lambda_mult,
        )

        selected = [search_response.results[i] for i in mmr_selected_ids]

        # TODO: Change result.id to text field + use k param
        return [Document(page_content=result.id) for result in selected]

    def _embed_query(self, query: str) -> List[float]:
        """Embed search query.

        Args:
            query: Query string to embed.

        Returns:
            List of floats representing the query embedding.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings function is not set")

        embedding = self.embeddings.embed_query(text=query)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        return embedding

    def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed search texts.

        Args:
            texts: Iterable of texts to embed.

        Returns:
            Nested list of floats representing the texts embeddings.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings function is not set")

        embeddings = self.embeddings.embed_documents(list(texts))
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        return embeddings

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        client: Optional[vantage.VantageClient] = None,
        collection: Optional[vantage.Collection] = None,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        user_provided_embeddings: Optional[bool] = False,
        llm: Optional[str] = None,
        external_key_id: Optional[str] = None,
    ) -> Vantage:
        """Return Vantage VectorStore initialized from raw text."""

        client = client or cls._get_vantage_client()

        if collection:
            vantage_vector_store = cls(
                client=client,
                embedding=embedding,
                collection=collection,
            )
        else:
            vantage_vector_store = cls(
                client=client,
                embedding=embedding,
                collection_id=collection_id,
                collection_name=collection_name,
                embedding_dimension=embedding_dimension,
                user_provided_embeddings=user_provided_embeddings,
                llm=llm,
                external_key_id=external_key_id,
            )

        vantage_vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        return vantage_vector_store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        client: Optional[vantage.VantageClient] = None,
        collection: Optional[vantage.Collection] = None,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        user_provided_embeddings: Optional[bool] = False,
        llm: Optional[str] = None,
        external_key_id: Optional[str] = None,
    ) -> Vantage:
        """Return Vantage VectorStore initialized from documents."""

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            ids=ids,
            client=client,
            collection=collection,
            collection_id=collection_id,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            user_provided_embeddings=user_provided_embeddings,
            llm=llm,
            external_key_id=external_key_id,
        )

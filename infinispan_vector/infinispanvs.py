"""Module providing Infinispan as a VectorStore"""

from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
import json
import uuid
import requests

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)
REST_TIMEOUT = 10


class Infinispan:
    """`Infinispan` REST interface.

    This class exposes the Infinispan operations needed to
    create and setup a vector db.

    You need a running Infinispan (15+) server without authentication.
    You can easily start one see: https://github.com/rigazilla/infinispan-vector
    """
    def __init__(
            self,
            configuration: Optional[dict[str, Any]] = None,
    ):
        self._configuration = configuration or {}
        self._schema = str(self._configuration.get("schema", "http"))
        self._host = str(self._configuration.get("hosts", ["127.0.0.1:11222"])[0])
        self._default_node = self._schema + "://" + self._host
        self._cache_url = str(self._configuration.get("cache_url", "/rest/v2/caches"))
        self._schema_url = str(self._configuration.get("cache_url", "/rest/v2/schemas"))
        self._use_post_for_query = str(self._configuration.get("use_post_for_query", True))

    def req_query(self, query, cache_name, local=False) -> requests.Response:
        """ Request a query
        Args:
            query(str): query requested
            cache_name(str): name of the target cache
            local(boolean): whether the query is local to clustered
        Returns:
            An http Response containing the result set or errors
        """
        if self._use_post_for_query:
            return self._req_query_post(query, cache_name, local)
        return self._req_query_get(query, cache_name, local)

    def _req_query_post(self, query_str, cache_name, local=False) -> requests.Response:
        api_url = (self._default_node + self._cache_url + "/" + cache_name
                   + "?action=search&local=" + str(local))
        data = {"query": query_str}
        data_json = json.dumps(data)
        response = requests.post(api_url, data_json, headers={"Content-Type": "application/json"},
                                 timeout=REST_TIMEOUT)
        return response

    def _req_query_get(self, query_str, cache_name, local=False) -> requests.Response:
        api_url = (self._default_node + self._cache_url + "/" + cache_name + "?action=search&query="
                   + query_str + "&local=" + str(local))
        response = requests.get(api_url, timeout=REST_TIMEOUT)
        return response

    def req_post(self, key, data, cache_name) -> requests.Response:
        """ Post an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.post(api_url, data, headers={"Content-Type": "application/json"},
                                 timeout=REST_TIMEOUT)
        return response

    def req_put(self, key: str, data, cache_name) -> requests.Response:
        """ Put an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.put(api_url, data, headers={"Content-Type": "application/json"},
                                timeout=REST_TIMEOUT)
        return response

    def req_get(self, key: str, cache_name) -> requests.Response:
        """ Get an entry
        Args:
            key(str): key of the entry
            cache_name(str): target cache
        Returns:
            An http Response containing the entry or errors
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.get(api_url, headers={"Content-Type": "application/json"},
                                timeout=REST_TIMEOUT)
        return response

    def req_schema_post(self, name, proto) -> requests.Response:
        """ Deploy a schema
        Args:
            name(str): name of the schema. Will be used as a key
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + "/" + name
        response = requests.post(api_url, proto, timeout=REST_TIMEOUT)
        return response

    def req_cache_post(self, name, config) -> requests.Response:
        """ Create a cache
        Args:
            name(str): name of the cache.
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + name
        response = requests.post(api_url, config, headers={"Content-Type": "application/json"},
                                 timeout=REST_TIMEOUT)
        return response

    def req_schema_delete(self, name) -> requests.Response:
        """ Delete a schema
        Args:
            name(str): name of the schema.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + "/" + name
        response = requests.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def req_cache_delete(self, name) -> requests.Response:
        """ Delete a cache
        Args:
            name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + name
        response = requests.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def req_cache_clear(self, cache_name) -> requests.Response:
        """ Clear a cache
        Args:
            name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "?action=clear"
        response = requests.post(api_url, timeout=REST_TIMEOUT)
        return response


class InfinispanVS(VectorStore):
    """`Infinispan` VectorStore interface.

        This class exposes the method to present Infinispan as a
        VectorStore

    Example:
        ... code-block:: python
            from langchain_community.vectorstores import InfinispanVS
            from mymodels import RGBEmbeddings
            ...
            vectorDb = InfinispanVS.from_documents(docs,
                            embedding=RGBEmbeddings(),
                            configuration={ "output_fields": ["texture", "color"],
                                            "lambda.key": lambda text,meta: str(meta["_key"]),
                                            "lambda.content": lambda item: item["color"]})

        or

        ... code-block:: python
            from langchain_community.vectorstores import InfinispanVS
            from mymodels import RGBEmbeddings
            ...
            ispn = Infinispan()
            # configure Infinispan here

            vectorDb = InfinispanVS.from_documents(docs,
                            embedding=RGBEmbeddings(),
                            configuration={ "output_fields": ["texture", "color"],
                                            "lambda.key": lambda text,meta: str(meta["_key"]),
                                            "lambda.content": lambda item: item["color"]},
                                            ispn = ispn)
    """

    def __init__(
            self,
            embedding,
            ispn: Infinispan,
            configuration: Optional[dict[str, Any]] = None,
    ):
        if ispn is not None:
            self.ispn = ispn
        else:
            self.ispn = Infinispan(configuration=configuration)
        self._configuration = configuration or {}
        self._cache_name = str(self._configuration.get("cache_name", "embeddingvectors"))
        self._entity_name = str(self._configuration.get("entity_name", "vector"))
        self._embedding = embedding
        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")
        self._get_key = configuration.get("lambda.key", lambda text, meta: str(uuid.uuid4()))
        self._to_content = configuration.get("lambda.content", lambda item: item["position"])
        self._output_fields = configuration.get("output_fields")

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None,
                  **kwargs: Any) -> List[str]:
        result = []
        embeds = self._embedding.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]
        data_input = list(zip(texts, metadatas, embeds))
        for text, metadata, embed in data_input:
            key = self._get_key(text, metadata)
            data = {"_type": self._entity_name, "floatVector": embed}
            data.update(metadata)
            data_str = json.dumps(data)
            self.ispn.req_put(key, data_str, self._cache_name)
            result.append(key)
        return result

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query."""
        documents = self.similarity_search_with_score(
            query=query, k=k
        )
        return [doc for doc, _ in documents]


    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.

        Returns:
            List[Tuple[Document, float]]
        """
        embed = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embed, k=k
        )
        return documents


    def similarity_search_by_vector(
            self, embedding, k=4, **kwargs) -> List[Document]:
        res = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc,_ in res]

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of pair (Documents, score) most similar to the query vector.
        """
        if self._output_fields is None:
            query_str = ("select v, score(v) from " + self._entity_name +
                         " v where v.floatVector <-> " + json.dumps(embedding) + "~" + str(k))
        else:
            query_proj = "select "
            for field in self._output_fields[:-1]:
                query_proj = query_proj+"v."+field+","
            query_proj = query_proj+self._output_fields[-1]
            query_str = (query_proj+", score(v) from " + self._entity_name +
                         " v where floatVector <-> " + json.dumps(embedding) + "~" + str(k))
        query_res = self.ispn.req_query(query_str, self._cache_name)
        result = json.loads(query_res.text)
        return self._query_result_to_docs(result)

    def _query_result_to_docs(self, result) -> List[Tuple[Document, float]]:
        documents = []
        for row in result["hits"]:
            hit = row["hit"] or {}
            doc = Document(page_content=self._to_content(hit))
            documents.append((doc, hit["__ISPN_Score"]))
        return documents

    @classmethod
    def from_texts(
            cls: Type[InfinispanVS],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            configuration: dict[str, Any] = None,
            **kwargs: Any
    ) -> InfinispanVS:
        """Return VectorStore initialized from texts and embeddings."""
        if not isinstance(kwargs.get("infinispan"), Infinispan):
            warnings.warn("infinispan input must be Infinispan object.")
        infinispan = cls(embedding=embedding, configuration=configuration,
                         ispn=kwargs.get("infinispan"))
        if texts:
            infinispan.add_texts(texts, metadatas)
        return infinispan

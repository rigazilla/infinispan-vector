from __future__ import annotations

import json
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import requests
import fjson


logger = logging.getLogger(__name__)


class Infinispan(VectorStore):
    """Wrapper around Infinispan (15+) as a vector database.
    Example:
        .. code-block:: python
            from langchain_community.vectorstores import Infinispan
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            ...
    """

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        result = []
        for text in texts:
            key = self._to_key(text)
            data = self._to_attributes(text)
            vec = self._embedding.embed_query(self._to_content(data))
            data["floatVector"] = vec
            dataStr = json.dumps(data)
            resp = self.req_put(key, dataStr)
            result.append(key)
        return result

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query."""
        embed = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embed, k=k
        )
        return [doc for doc, _ in documents]


    def __init__(
        self,
        embedding,
        configuration: Optional[dict[str, Any]] = {"hosts" : ["127.0.0.1:11222"]},
    ):
        self._configuration = configuration
        self._schema = str(self._configuration.get("schema","http"))
        self._host = str(self._configuration.get("hosts",["127.0.0.1:11222"])[0])
        self._cache_url = str(self._configuration.get("cache_url","/rest/v2/caches"))
        self._schema_url = str(self._configuration.get("cache_url","/rest/v2/schemas"))
        self._cache_name = str(self._configuration.get("cache_name","embeddingvectors"))
        self._entity_name = str(self._configuration.get("entity_name","vector"))
        self._use_post_for_query = str(self._configuration.get("use_post_for_query",True))
        self._default_node = self._schema+"://"+self._host
        self._embedding = embedding
        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")
        # self._to_content = lambda hit: hit["content"] or {}
        self._to_key = configuration.get("lambda.key",lambda text: str(text["_key"]))
        self._to_attributes = configuration.get("lambda.attributes",lambda item: {x:item[x] for x in item if x != '_key'})
        self._to_content =  configuration.get("lambda.content", lambda item: item["position"])

    def similarity_search_by_vector(
        self, embedding, k = 4, **kwargs)  -> List[Document]:
        query_str = "from "+self._entity_name+"v where v.floatVector <-> "+json.dumps(embedding)+"~"+str(k)
        query_res = self.req_query(query_str)
        result = json.loads(query_res.text)
        return self.query_to_document(result)

    def similarity_search_with_score_by_vector(
                  self, embedding: List[float], k: int = 4, **kwargs: Any
              ) -> List[Tuple[Document, float]]:
        # Workaround, trunc float for http/get
        query_str = "from "+self._entity_name+" v where v.floatVector <-> "+json.dumps(embedding)+"~"+str(k)
        query_res = self.req_query(query_str)
        result = json.loads(query_res.text)
        return self.query_to_document(result)

    def query_to_document(self, result) -> List[Document]:
        documents = []
        for row in result["hits"]:
            hit = row["hit"] or {}
            doc = Document(page_content=self._to_content(hit))
            documents.append((doc, 0))
        return documents

    def req_query(self, query_str, local = False) -> requests.Response:
        if self._use_post_for_query:
            return self.req_query_post(query_str, local)
        else:
            return self.req_query_get(query_str, local)


    def req_query_post(self, query_str, local = False) -> requests.Response:
        api_url=self._default_node+self._cache_url+"/"+self._cache_name+"?action=search"
        data = { "query" : query_str }
        dataJson = json.dumps(data)
        response = requests.post(api_url, dataJson, headers= {"Content-Type" : "application/json"})
        return response

    def req_query_get(self, query_str, local = False) -> requests.Response:
        api_url=self._default_node+self._cache_url+"/"+self._cache_name+"?action=search&query="+query_str+"&local="+str(local)
        response = requests.get(api_url)
        return response


    def req_post(self, key, data, cache_name = None) -> requests.Response:
        if cache_name == None:
            cache_name = self._cache_name
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.post(api_url, data, headers= {"Content-Type" : "application/json"})
        return response

    def req_put(self, key:str, data, cache_name = None) -> requests.Response:
        if cache_name == None:
            cache_name = self._cache_name
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = requests.put(api_url, data, headers= {"Content-Type" : "application/json"})
        return response

    def req_schema_post(self, name, proto) -> requests.Response:
        api_url = self._default_node + self._schema_url + "/" + name
        response = requests.post(api_url, proto)
        return response

    def req_cache_post(self, name, config) -> requests.Response:
        api_url = self._default_node + self._cache_url + "/" + name
        response = requests.post(api_url, config, headers= {"Content-Type" : "application/json"})
        return response

    def req_schema_delete(self, name) -> requests.Response:
        api_url = self._default_node + self._schema_url + "/" + name
        response = requests.delete(api_url)
        return response

    def req_cache_delete(self, name) -> requests.Response:
        api_url = self._default_node + self._cache_url + "/" + name
        response = requests.delete(api_url)
        return response

    def req_cache_clear(self) -> requests.Response:
        api_url = self._default_node + self._cache_url + "/" + self._cache_name + "?action=clear"
        response = requests.post(api_url)
        return response

    @classmethod
    def from_texts(
        cls: Type[Infinispan],
        embedding: Embeddings,
        configuration: dict[str, Any] = {"hosts" : ["127.0.0.1:11222"]}
    ) -> Infinispan:
        """Return VectorStore initialized from texts and embeddings."""
        infinispan = cls(embedding=embedding, configuration=configuration)
        return infinispan

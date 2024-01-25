import json
from typing import List, Optional

from langchain_core.documents import Document
import requests
from infinispan_vector import Infinispan
from tests.integration_tests.vectorstores.fake_embeddings import (
    RGBEmbeddings
)

def _infinispan_from_texts() -> Infinispan:
    return Infinispan.from_texts({}, embedding=RGBEmbeddings(),
                                 configuration={"lambda.content": lambda item: item["color"]})

def test_infinispan_schema_post() -> None:
    test_infinispan_schema_delete()
    infinispan = _infinispan_from_texts()
    data = '''
/**
 * @Indexed
 */
message vector {
/**
 * @Vector(dimension=3)
 */
repeated float floatVector = 1;
optional string texture = 2;
optional string color = 3;
optional string _key = 4;
}
'''
    output = infinispan.req_schema_post("vector.proto",data)
    assert output.status_code == 200
    assert json.loads(output.text)["error"] == None

def test_infinispan_schema_delete() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_cache_clear()
    output = infinispan.req_schema_delete("vector.proto")
    assert output.status_code in {204, 404}

def test_cache_create_delete() -> None:
    _cache_delete()
    _cache_post()
    _cache_delete()

def _cache_post() -> None:
    infinispan = _infinispan_from_texts()
    data = '''
{
  "distributed-cache": {
    "owners": "2",
    "mode": "SYNC",
    "statistics": true,
    "encoding": {
      "media-type": "application/x-protostream"
    },
    "indexing": {
      "enabled": true,
      "storage": "filesystem",
      "startup-mode": "AUTO",
      "indexing-mode": "AUTO",
      "indexed-entities": [
        "vector"
      ]
    }
  }
}
'''
    output = infinispan.req_cache_post("embeddingvectors",data)
    assert (output.status_code == 200 or ("already exists" in output.text))

def _cache_delete() -> None:
    infinispan = _infinispan_from_texts()
    output = infinispan.req_cache_delete("embeddingvectors")
    assert output.status_code in {200,404}

def test_infinispan_vectors_post_then_query() -> None:
    infinispan = _infinispan_from_texts()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear()
    test_add_texts()
    query_res = infinispan.req_query("from vector i where i.floatVector <-> [0.5,0.5,0.5]~2")
    assert query_res.status_code == 200
    str = query_res.content.decode("utf-8")
    jOut = json.loads(str)
    assert jOut["hit_count"] == 2
    vectors = [x["hit"]["color"] for x in jOut["hits"]]
    vectors.sort()
    assert vectors == ['green', 'red']

def test_infinispan_vectors_post_then_query_by_vector() -> None:
    infinispan = _infinispan_from_texts()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear()
    test_add_texts()
    docs = infinispan.similarity_search_by_vector([0.5,0.5,0.0],2)
    assert docs == [(Document(page_content='green'), 0), (Document(page_content='red'), 0)]

def test_infinispan_similarity_search() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_cache_clear()
    test_add_texts()
    output = infinispan.similarity_search("orange",2)
    assert output == [Document(page_content='red'), Document(page_content='green')]
    output = infinispan.similarity_search("purple",2)
    assert output == [Document(page_content='red'), Document(page_content='blue')]
    output = infinispan.similarity_search("lime",2)
    assert output == [Document(page_content='green'), Document(page_content='black')]
    output = infinispan.similarity_search("snow",2)
    assert output == [Document(page_content='white'), Document(page_content='red')]

def test_put_get() -> None:
    infinispan = _infinispan_from_texts()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear()
    metadata = {"_type": "vector", "_key": "1", "texture" : "matt", "color" : "red", "floatVector" : [0.0,0.0,1.0]}
    infinispan.req_put("1", json.dumps(metadata))
    response = infinispan.req_get("1")
    res = json.loads(response.text)
    assert res==metadata

def test_add_texts() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_cache_clear()
    metadatas = [{"_key": 1, "_type": "vector", "texture" : "matt", "color" : "red"},
                {"_key": 2, "_type": "vector", "texture": "glossy", "color" : "green"},
                {"_key": 3, "_type": "vector", "texture": "silk", "color" : "blue"},
                {"_key": 4, "_type": "vector", "texture": "matt", "color" : "black"},
                {"_key": 5, "_type": "vector", "texture": "raw", "color" : "white"}
            ]
    texts = ["red", "green", "blue", "black", "white"]
    res = infinispan.add_texts(texts, metadatas)
    assert res == ["1","2","3","4","5"]

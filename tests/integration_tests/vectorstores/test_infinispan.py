import json
from typing import List, Optional

from langchain_core.documents import Document
import requests
from infinispan_vector import Infinispan, InfinispanVS
from infinispan_vector.infinispanvs import Infinispan
from fake_embeddings import (
    RGBEmbeddings
)

cache_name="embeddingvectors"
def _get_ispn() -> (Infinispan, InfinispanVS):
    ispn = Infinispan()
    return (ispn, InfinispanVS.from_texts({}, embedding=RGBEmbeddings(),
                                 configuration={"output_fields": ["texture", "color"], "lambda.key": lambda text, meta: str(meta["_key"]), "lambda.content": lambda item: item["color"]}, ispn=ispn))

def test_infinispan_schema_post() -> None:
    test_infinispan_schema_delete()
    infinispan, vs = _get_ispn()
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
    infinispan, vs = _get_ispn()
    infinispan.req_cache_clear(cache_name)
    output = infinispan.req_schema_delete("vector.proto")
    assert output.status_code in {204, 404}

def test_cache_create_delete() -> None:
    _cache_delete()
    _cache_post()
    _cache_delete()

def _cache_post() -> None:
    infinispan, vs = _get_ispn()
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
    infinispan, vs = _get_ispn()
    output = infinispan.req_cache_delete("embeddingvectors")
    assert output.status_code in {200,404}

def test_infinispan_vectors_post_then_query() -> None:
    infinispan, vs = _get_ispn()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear(cache_name)
    test_add_texts()
    query_res = infinispan.req_query("from vector i where i.floatVector <-> [0.5,0.5,0.5]~2", cache_name)
    assert query_res.status_code == 200
    str = query_res.content.decode("utf-8")
    jOut = json.loads(str)
    assert jOut["hit_count"] == 2
    vectors = [x["hit"]["color"] for x in jOut["hits"]]
    vectors.sort()
    assert vectors == ['green', 'red']

def test_infinispan_vectors_post_then_query_with_score() -> None:
    infinispan, vs = _get_ispn()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear(cache_name)
    test_add_texts()
    query_res = infinispan.req_query("select i.texture, i.color, score(i) from vector i where i.floatVector <-> [0.5,0.5,0.5]~2", cache_name)
    assert query_res.status_code == 200
    str = query_res.content.decode("utf-8")
    jOut = json.loads(str)
    assert jOut["hit_count"] == 2
    vectors = [x["hit"]["color"] for x in jOut["hits"]]
    vectors.sort()
    assert vectors == ['green', 'red']

def test_infinispan_vectors_post_then_query_by_vector() -> None:
    infinispan, vs = _get_ispn()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear(cache_name)
    test_add_texts()
    docs = vs.similarity_search_by_vector([0.5,0.5,0.0],2)
    assert docs == [Document(page_content='green'), Document(page_content='red')]

def test_infinispan_similarity_search() -> None:
    infinispan, vs = _get_ispn()
    infinispan.req_cache_clear(cache_name)
    test_add_texts()
    output = vs.similarity_search("orange",2)
    assert output == [Document(page_content='red'), Document(page_content='green')]
    output = vs.similarity_search("purple",2)
    assert output == [Document(page_content='red'), Document(page_content='blue')]
    output = vs.similarity_search("lime",2)
    assert output == [Document(page_content='green'), Document(page_content='black')]
    output = vs.similarity_search("snow",2)
    assert output == [Document(page_content='white'), Document(page_content='red')]

def test_infinispan_similarity_search_with_score() -> None:
    infinispan, vs = _get_ispn()
    infinispan.req_cache_clear(cache_name)
    test_add_texts()
    output = vs.similarity_search("orange",2)
    assert output == [Document(page_content='red'), Document(page_content='green')]
    output = vs.similarity_search("purple",2)
    assert output == [Document(page_content='red'), Document(page_content='blue')]
    output = vs.similarity_search("lime",2)
    assert output == [Document(page_content='green'), Document(page_content='black')]
    output = vs.similarity_search("snow",2)
    assert output == [Document(page_content='white'), Document(page_content='red')]

def test_put_get() -> None:
    infinispan, vs = _get_ispn()
    test_infinispan_schema_post()
    _cache_post()
    infinispan.req_cache_clear(cache_name)
    metadata = {"_type": "vector", "_key": "1", "texture" : "matt", "color" : "red", "floatVector" : [0.0,0.0,1.0]}
    infinispan.req_put("1", json.dumps(metadata), cache_name)
    response = infinispan.req_get("1", cache_name)
    res = json.loads(response.text)
    assert res==metadata

def test_add_texts() -> None:
    infinispan, vs = _get_ispn()
    infinispan.req_cache_clear(cache_name)
    metadatas = [{"_key": 1, "_type": "vector", "texture" : "matt", "color" : "red"},
                {"_key": 2, "_type": "vector", "texture": "glossy", "color" : "green"},
                {"_key": 3, "_type": "vector", "texture": "silk", "color" : "blue"},
                {"_key": 4, "_type": "vector", "texture": "matt", "color" : "black"},
                {"_key": 5, "_type": "vector", "texture": "raw", "color" : "white"}
            ]
    texts = ["red", "green", "blue", "black", "white"]
    res = vs.add_texts(texts, metadatas)
    assert res == ["1","2","3","4","5"]

from typing import Any
from langchain_core.vectorstores import VectorStore

def _import_infinispan() -> Any:
    from infinispan_vector.infinispan import Infinispan
    return Infinispan

def __getattr__(name: str) -> Any:
    if name == "Infinispan":
        return _import_infinispan()
__all__ = [
    "Infinispan"
]
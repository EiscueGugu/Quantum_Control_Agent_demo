"""RAG 检索器：根据查询从 Chroma 检索相关知识。"""

from typing import List

import logging
import os

# 抑制 HuggingFace / sentence-transformers 的加载日志
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
for name in ["sentence_transformers", "transformers", "huggingface_hub", "tqdm"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class RAGRetriever:
    def __init__(self, persist_directory: str = "data/chroma_db_v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
            self._ready = True
        except Exception:
            self._ready = False
            self.vectorstore = None

    def search(self, query: str, k: int = 3) -> List[str]:
        if not self._ready or self.vectorstore is None:
            return ["(知识库未初始化，请先运行 RAGIndexer().build())"]
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    @property
    def is_ready(self) -> bool:
        return self._ready

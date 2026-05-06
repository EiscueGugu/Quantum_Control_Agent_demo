"""RAG 索引构建：加载文档 → 切分 → 向量化 → 存入 Chroma。"""

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
for name in ["sentence_transformers", "transformers", "huggingface_hub", "tqdm"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGIndexer:
    def __init__(self, persist_directory: str = "data/chroma_db_v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.persist_directory = persist_directory
        self.vectorstore = None

    def load_and_split_documents(self, data_dir: str = "knowledge_base"):
        docs = []
        if not os.path.isdir(data_dir):
            return docs

        for filename in os.listdir(data_dir):
            path = os.path.join(data_dir, filename)
            if not os.path.isfile(path):
                continue
            if filename.endswith((".txt", ".md")):
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n", "。", ".", " "],
        )
        return splitter.split_documents(docs)

    def build(self, data_dir: str = "knowledge_base"):
        docs = self.load_and_split_documents(data_dir)
        if not docs:
            print("没有找到文档，跳过索引构建。")
            return

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"RAG 索引构建完成: {len(docs)} 个文本块 → {self.persist_directory}")

    def load(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        return self

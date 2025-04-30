from typing import List, Any, Optional, Dict, Tuple ,TypedDict, Annotated
import os

from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

import config


def build_k_notion(notion_folder: str, index_folder: str):
    loader = NotionDirectoryLoader(notion_folder, encoding="utf-8")
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    emb_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vs = FAISS.from_documents(documents=splits, embedding=emb_model)

    vs.save_local(index_folder)



if __name__ == "__main__":
    index_folder = config.NOTION_INDEX_FOLDER 
    build_k_notion("Notion0code", index_folder)
import logging

from typing import List, Any, Optional, Dict, Tuple, TypedDict, Annotated
import os, torch, pickle


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.storage import InMemoryByteStore

from agents.retrievers.cross_encoder_reranker_with_score import CrossEncoderRerankerWithScores
from agents.retrievers.utils.models_builder import (
    getEmbeddingModel,
    getRerankerModel,
)

from agents.retrievers.teamly_retriever import (
    TeamlyRetriever,
    TeamlyRetriever_Tickets,
    TeamlyRetriever_Glossary,
    TeamlyContextualCompressionRetriever,
)

import config


_teamly_retriever_instance: Optional[TeamlyRetriever] = None
#_teamly_retriever_tickets_instance : Optional[TeamlyRetriever_Tickets] = None
#_teamly_retriever_glossary_instance : Optional[TeamlyRetriever_Glossary] = None

_teamly_reranker_retriever: Optional[TeamlyContextualCompressionRetriever] = None
_faiss_reranker_retriever: Optional[ContextualCompressionRetriever] = None

_faiss_indexes = {}
_multi_retrievers = {}


def getFAISSIndex(file_path: str)-> FAISS:
    global _faiss_indexes
    index = _faiss_indexes.get(file_path, None)
    if index is None:
        index = FAISS.load_local(file_path, getEmbeddingModel(), allow_dangerous_deserialization=True)
        _faiss_indexes[file_path] = index
    return index

def load_vectorstore(file_path: str) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")
    return getFAISSIndex(file_path)

def buildEnsembleRetriever(index_paths: list[str], search_kwargs: dict, weights: list[float])-> EnsembleRetriever:
    base_retrievers = []
    for index_path in index_paths:
        base_retrievers.extend(load_vectorstore(index_paths).as_retriever(search_kwargs=search_kwargs))
    return EnsembleRetriever(
        retrievers=[base_retrievers],
        weights=weights  # adjust to favor text vs. images
    )

def buildMultiRetriever(index_paths: list[str], search_kwargs: dict, weights: list[float])-> ContextualCompressionRetriever:
    global _multi_retrievers
    retriever = _multi_retrievers.get(index_paths, None)
    if retriever is None:
        ensemble = buildEnsembleRetriever(index_paths, search_kwargs, weights)
        reranker_model = getRerankerModel()
        reranker = CrossEncoderRerankerWithScores(model=reranker_model, top_n=3, min_ratio=float(config.MIN_RERANKER_RATIO))
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble
        )
        _multi_retrievers[index_paths] = retriever
    return retriever

_MAX_TEAMLY_RETRIEVALS = 40
_MAX_RETRIEVALS = 3

def buildTeamlyRetriever()-> TeamlyContextualCompressionRetriever:
    global _teamly_retriever_instance
    global _teamly_reranker_retriever

    if _teamly_reranker_retriever is None:
        # Initialize Teamly retriever with refresh support
        if _teamly_retriever_instance is None:
            _teamly_retriever_instance = TeamlyRetriever("./auth.json", k=_MAX_TEAMLY_RETRIEVALS)
        reranker_model = getRerankerModel()

        reranker = CrossEncoderRerankerWithScores(
            model=reranker_model, top_n=_MAX_RETRIEVALS, 
            min_ratio=float(config.MIN_RERANKER_RATIO)
        )
        _teamly_reranker_retriever = TeamlyContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=_teamly_retriever_instance
        )
    return _teamly_reranker_retriever

def buildFAISSRetriever()-> ContextualCompressionRetriever:
    global _faiss_reranker_retriever
    if _faiss_reranker_retriever is None:
        vector_store_path = config.ASSISTANT_INDEX_FOLDER
        vectorstore = load_vectorstore(vector_store_path)

        with open(f'{vector_store_path}/docstore.pkl', 'rb') as file:
            documents = pickle.load(file)
        doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]

        store = InMemoryByteStore()
        id_key = "problem_number"
        multi_retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": _MAX_RETRIEVALS},
        )
        multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
        reranker_model = getRerankerModel()
        reranker = CrossEncoderRerankerWithScores(
            model=reranker_model, 
            top_n=_MAX_RETRIEVALS, 
            min_ratio=float(config.MIN_RERANKER_RATIO)
        )
        _faiss_reranker_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=multi_retriever
        )

    return _faiss_reranker_retriever


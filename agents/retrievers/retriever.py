import logging

from typing import List, Any, Optional, Dict, Tuple, TypedDict, Annotated
import os
import pickle
import torch
from langchain_community.document_loaders import NotionDBLoader
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.docstore.document import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import InMemoryByteStore
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_community.vectorstores import FAISS
from palimpsest import Palimpsest
from agents.retrievers.teamly_retriever import (
    TeamlyRetriever,
    TeamlyRetriever_Tickets,
    TeamlyRetriever_Glossary,
    TeamlyContextualCompressionRetriever
)
import config

# Global instances and refreshable Teamly Retriever for hot index updates
_teamly_retriever_instance: Optional[TeamlyRetriever] = None
_teamly_retriever_tickets_instance : Optional[TeamlyRetriever_Tickets] = None
_teamly_retriever_glossary_instance : Optional[TeamlyRetriever_Glossary] = None
#_teamly_compression_retriever_instance: Optional[TeamlyContextualCompressionRetriever] = None

def load_vectorstore(file_path: str, embedding_model_name: str) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

def get_retriever_multi():
    notion_vs = load_vectorstore(config.NOTION_INDEX_FOLDER, config.EMBEDDING_MODEL)
    chats_vs = load_vectorstore(config.CHATS_INDEX_FOLDER, config.EMBEDDING_MODEL)
    k = 5
    ensemble = EnsembleRetriever(
        retrievers=[notion_vs.as_retriever(search_kwargs={"k": k}),
                    chats_vs.as_retriever(search_kwargs={"k": k})],
        weights=[0.5, 0.5]  # adjust to favor text vs. images
    )
    reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL)
    reranker = CrossEncoderReranker(model=reranker_model, top_n=3)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )
    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": k})
        # docs = retriever.similarity_search_with_score(query, k=5)
        # result = [doc for doc, score in docs if score >= 0.20]
        return result
    return search

def get_retriever_teamly():
    MAX_RETRIEVALS = 3
    global _teamly_retriever_instance#, _teamly_compression_retriever_instance
    # Initialize Teamly retriever with refresh support
    _teamly_retriever_instance = TeamlyRetriever("./auth.json", k=40)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    reranker_model = HuggingFaceCrossEncoder(
        model_name=config.RERANKING_MODEL,
        model_kwargs={'trust_remote_code': True, "device": device}
    )
    reranker = CrossEncoderReranker(model=reranker_model, top_n=MAX_RETRIEVALS)
    retriever = TeamlyContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=_teamly_retriever_instance
    )
    def search(query: str) -> List[Document]:
        try:
            result = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        except Exception as e:
            logging.error("Error occured during teamly search tool calling.\nException: {e}")
            raise e
        # torch.cuda.empty_cache()
        return result
    return search

def get_retriever_faiss():
    MAX_RETRIEVALS = 3
    vector_store_path = config.ASSISTANT_INDEX_FOLDER
    vectorstore = load_vectorstore(vector_store_path, config.EMBEDDING_MODEL)
    with open(f'{vector_store_path}/docstore.pkl', 'rb') as file:
        documents = pickle.load(file)
    doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
    store = InMemoryByteStore()
    id_key = "problem_number"
    multi_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": MAX_RETRIEVALS},
    )
    multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_model = HuggingFaceCrossEncoder(
        model_name=config.RERANKING_MODEL,
        model_kwargs={'trust_remote_code': True, "device": device}
    )
    reranker = CrossEncoderReranker(model=reranker_model, top_n=MAX_RETRIEVALS)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=multi_retriever
    )
    def search(query: str) -> List[Document]:
        try:
            result = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        except Exception as e:
            logging.error("Error occured during faiss search tool calling.\nException: {e}")
            raise e
        return result
    return search

def get_retriever():
    retriever_type = config.RETRIEVER_TYPE
    if retriever_type == "teamly":
        return get_retriever_teamly()
    return get_retriever_faiss()

# Initialize the search function with the selected retrieverx
search = get_retriever()


def refresh_indexes():
    """Refresh the indexes of the active retriever (e.g., rebuild Teamly FAISS and BM25 indexes)."""
    logging.info("Refreshing faiss indexes...")
    if config.RETRIEVER_TYPE == "teamly" and _teamly_retriever_instance:
        _teamly_retriever_instance.refresh()
    if _teamly_retriever_tickets_instance:
        _teamly_retriever_tickets_instance.refresh()
    logging.info("...complete refreshing faiss indexes.")

def get_search_tool(anonymizer: Palimpsest = None):
    @tool
    def search_kb(query: str) -> str:
        """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question
        Returns:
            Context from knowledgebase suitable for the query.
        """
        found_docs = search(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return search_kb

def get_tickets_search_tool(anonymizer: Palimpsest = None):
    MAX_RETRIEVALS = 3
    global _teamly_retriever_tickets_instance

    _teamly_retriever_tickets_instance = TeamlyRetriever_Tickets("./auth_tickets.json", k=MAX_RETRIEVALS)
    
    @tool
    def search_tickets(query: str) -> str:
        """Retrieves from tickets knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question
        Returns:
            Context from knowledgebase suitable for the query.
        """
        found_docs = _teamly_retriever_tickets_instance.invoke(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return search_tickets

def get_term_and_defition_tools(anonymizer: Palimpsest = None):
    MAX_RETRIEVALS = 3
    global _teamly_retriever_glossary_instance

    _teamly_retriever_glossary_instance = TeamlyRetriever_Glossary("./auth_glossary.json", k=MAX_RETRIEVALS)
    
    @tool
    def lookup_term(term: str) -> str:
        """
        Look up the definition of a term or abbreviation in the reference dictionary.

        This tool is designed to retrieve the meaning of either a full term 
        or an abbreviation from a predefined reference source. 
        All abbreviations in the reference are stored in uppercase. 
        All terms in the reference are stored in singular nominative case. 

        The input must strictly follow these conventions:
        - Abbreviations: uppercase only (e.g., "HTTP", "NASA", "АД").
        - Terms: singular nominative case (e.g., "server", "network", "лизинговая заявка").

        Args:
            name (str): The term or abbreviation to look up.
                Must match the format and casing conventions of the reference.

        Returns:
            str: The definition or description of the provided term or abbreviation.
                Currently returns a constant placeholder string.
        """
        found_docs = _teamly_retriever_glossary_instance.invoke(term)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return lookup_term

if __name__ == '__main__':
    search_kb = get_search_tool()
    answer = search_kb("Кто такие кей юзеры?")
    print(answer)
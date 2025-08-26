import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import config


_device = "cuda" if torch.cuda.is_available() else "cpu"


_embedding_model: HuggingFaceEmbeddings = None
_reranker_model: HuggingFaceCrossEncoder = None

def getEmbeddingModel()-> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_model

def getRerankerModel()-> HuggingFaceCrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = HuggingFaceCrossEncoder(
            model_name=config.RERANKING_MODEL, 
            model_kwargs={'trust_remote_code': True, "device": _device}
        )
    return _reranker_model
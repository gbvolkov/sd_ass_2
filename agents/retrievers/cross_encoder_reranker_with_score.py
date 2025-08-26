
from copy import deepcopy

from langchain.retrievers.document_compressors import CrossEncoderReranker

class CrossEncoderRerankerWithScores(CrossEncoderReranker):
    min_ratio: int = 0

    def __init__(self, *args, min_ratio: float = 0.00, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_ratio=min_ratio
    def compress_documents(self, documents, query, callbacks=None):
        # compute scores
        scores = self.model.score([(query, d.page_content) for d in documents])
        # attach to metadata (without mutating originals)
        docs = []
        for d, s in zip(documents, scores):
            d2 = deepcopy(d)
            d2.metadata = {**(d2.metadata or {}), "rerank_score": float(s)}
            docs.append(d2)
        max_s = max(scores)
        threshold = self.min_ratio*max_s
        passed_docs = [d for d in docs if d.metadata["rerank_score"] >= threshold]
        # sort by score desc and keep top_n
        passed_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return passed_docs[: self.top_n]
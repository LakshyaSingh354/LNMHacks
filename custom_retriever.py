from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

import torch
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from typing import List

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search with BERT reranking."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        bert_model: str = "nlpaueb/legal-bert-base-uncased",
        mode: str = "AND",
    ) -> None:
        """Initialize retriever with vector, keyword retrievers, and BERT for reranking."""
        
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode. Must be 'AND' or 'OR'.")
        self._mode = mode
        
        # Load the BERT model for reranking
        self.tokenizer = AutoTokenizer.from_pretrained("onnx/")

        self.model = ORTModelForSequenceClassification.from_pretrained("onnx/", file_name='model_quantized.onnx')
        
        super().__init__()

    def _compute_bert_score(self, query: str, document_text: str) -> float:
        """Compute the BERT score for ranking a query-document pair."""
        inputs = self.tokenizer(query, document_text, return_tensors='pt', truncation=True, max_length=512)
        outputs = self.model(**inputs)
        score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Higher score means more relevant
        return score

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query using both vector and keyword retrievers, then rerank with BERT."""
        
        # Perform initial retrieval with vector and keyword retrievers
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        
        # Create a dictionary to combine nodes from both retrievers
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})
        
        # Decide which nodes to return based on the mode
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)
        
        # Get the combined nodes to be reranked
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        # Rerank the retrieved nodes using BERT
        reranked_nodes = self._rerank_with_bert(query_bundle.query_str, retrieve_nodes)
        
        return reranked_nodes

    def _rerank_with_bert(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Rerank nodes using BERT scores."""
        scored_nodes = []
        for node in nodes:
            bert_score = self._compute_bert_score(query, node.node.text)
            scored_nodes.append((node, bert_score))
        
        # Sort nodes by BERT score in descending order
        scored_nodes = sorted(scored_nodes, key=lambda x: x[1], reverse=True)
        
        # Return the nodes sorted by relevance
        return [node for node, score in scored_nodes]
import streamlit as st
import logging
from typing import List, Any, Dict, Optional
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
from langchain.schema import BaseRetriever, Document
from pydantic import Field
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

class CustomOllama(Ollama):
    class Config:
        extra = "allow"

class RerankingRetriever(BaseRetriever):
    vectorstore: Any
    initial_k: int = 10
    final_k: int = 5
    # GPU device changed to 1
    cross_encoder: Any = Field(default_factory=lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda:1"))

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        candidate_docs = self.vectorstore.similarity_search(query, k=self.initial_k)
        pairs = [(query, doc.page_content) for doc in candidate_docs]

        # Move the cross_encoder model to GPU for fast inference, if available.
        if torch.cuda.is_available():
            # GPU device changed to 1
            self.cross_encoder.model.to("cuda:1")

        scores = self.cross_encoder.predict(pairs)

        # Offload the model to CPU to free up GPU memory.
        self.cross_encoder.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Debug: Log candidate scores and associated document metadata.
        print(f"[Rerank] Query: {query}")
        for i, (score, doc) in enumerate(zip(scores, candidate_docs)):
            print(f"[Rerank] Candidate {i}: Score {score}, Source: {doc.metadata.get('source', 'N/A')}")

        ranked_docs = [doc for score, doc in sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)]

        print("[Rerank] Final ranked documents (top k):")
        for i, doc in enumerate(ranked_docs[:self.final_k]):
            print(f"[Rerank] Rank {i+1}: Metadata: {doc.metadata}")
            print(f"[Rerank] Rank {i+1}: Content snippet: {doc.page_content[:100]}...")
        return ranked_docs[:self.final_k]

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Async version simply calls the synchronous method.
        return self.get_relevant_documents(query, **kwargs)

class BalancedMultimodalRetriever(RerankingRetriever):
    """Extends RerankingRetriever to balance results across modalities."""
    
    modality_distribution: Optional[Dict[str, int]] = None
    target_distribution: Optional[Dict[str, float]] = None
    
    def __init__(self, vectorstore, initial_k=10, final_k=5, modality_distribution=None, **kwargs):
        super().__init__(vectorstore=vectorstore, initial_k=initial_k, final_k=final_k, **kwargs)
        self.modality_distribution = modality_distribution or {}
        
        # Calculate target distribution percentages
        total_docs = sum(self.modality_distribution.values())
        if total_docs > 0:
            self.target_distribution = {
                modality: min(max(0.2, count / total_docs), 0.7)  # Minimum 20%, maximum 70%
                for modality, count in self.modality_distribution.items()
                if count > 0
            }
        else:
            self.target_distribution = {}
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # First get documents using the base similarity search
        candidate_docs = self.vectorstore.similarity_search(query, k=self.initial_k)
        
        # If no modality balancing is needed, use standard reranking
        if not self.target_distribution:
            return super().get_relevant_documents(query, **kwargs)
        
        # Group documents by modality
        grouped_docs = {}
        for doc in candidate_docs:
            modality = doc.metadata.get("type", "text")  # Default to text
            if modality not in grouped_docs:
                grouped_docs[modality] = []
            grouped_docs[modality].append(doc)
        
        # Calculate how many docs to include from each modality
        modality_counts = {}
        for modality, target_pct in self.target_distribution.items():
            # Calculate target count but ensure at least 1 if available
            if modality in grouped_docs and len(grouped_docs[modality]) > 0:
                target_count = max(1, int(self.final_k * target_pct))
                modality_counts[modality] = min(target_count, len(grouped_docs[modality]))
        
        # Adjust counts to ensure we get exactly final_k docs
        total_allocated = sum(modality_counts.values())
        
        # If we're short, add more from modalities with extra docs
        if total_allocated < self.final_k:
            deficit = self.final_k - total_allocated
            for modality in sorted(grouped_docs.keys(), 
                                  key=lambda m: len(grouped_docs[m]) - modality_counts.get(m, 0),
                                  reverse=True):
                available = len(grouped_docs[modality]) - modality_counts.get(modality, 0)
                to_add = min(deficit, available)
                if to_add > 0:
                    modality_counts[modality] = modality_counts.get(modality, 0) + to_add
                    deficit -= to_add
                if deficit == 0:
                    break
        
        # If we still need more, add docs from modalities not in the target
        if total_allocated < self.final_k:
            for modality in grouped_docs:
                if modality not in modality_counts and len(grouped_docs[modality]) > 0:
                    modality_counts[modality] = min(self.final_k - sum(modality_counts.values()),
                                                   len(grouped_docs[modality]))
                    if sum(modality_counts.values()) >= self.final_k:
                        break
        
        # Create the balanced result set
        results = []
        for modality, count in modality_counts.items():
            if modality in grouped_docs and count > 0:
                # Apply reranking within each modality group using cross-encoder
                if len(grouped_docs[modality]) > count:
                    pairs = [(query, doc.page_content) for doc in grouped_docs[modality]]
                    
                    # Move the cross_encoder model to GPU for fast inference, if available.
                    if torch.cuda.is_available():
                        # GPU device changed to 1
                        self.cross_encoder.model.to("cuda:1")
                    
                    scores = self.cross_encoder.predict(pairs)
                    
                    # Offload the model to CPU to free up GPU memory.
                    self.cross_encoder.model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    ranked_docs = [doc for score, doc in 
                                 sorted(zip(scores, grouped_docs[modality]), 
                                       key=lambda x: x[0], reverse=True)]
                    results.extend(ranked_docs[:count])
                else:
                    results.extend(grouped_docs[modality][:count])
        
        # Log the balanced retrieval results
        print(f"[BalancedRetriever] Query: {query}")
        print(f"[BalancedRetriever] Total candidates: {len(candidate_docs)}")
        for modality, docs in grouped_docs.items():
            print(f"[BalancedRetriever] {modality}: {len(docs)} candidates, {modality_counts.get(modality, 0)} selected")
        
        return results[:self.final_k]
    
    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Async version simply calls the synchronous method.
        return self.get_relevant_documents(query, **kwargs)

def create_balanced_retriever(vectorstore, initial_k=10, final_k=5, modality_distribution=None):
    """
    Create a retriever that balances results across different modalities.
    
    Args:
        vectorstore: The vector store to retrieve from
        initial_k: Number of candidates to retrieve initially
        final_k: Number of results to return after reranking
        modality_distribution: Dict with counts of each modality type
        
    Returns:
        A balanced multimodal retriever
    """
    return BalancedMultimodalRetriever(
        vectorstore=vectorstore,
        initial_k=initial_k,
        final_k=final_k,
        modality_distribution=modality_distribution
    )

def setup_qa_chain(vectorstore, output_tokens: int, temperature: float, balance_multimodal: bool = False, modality_distribution=None):
    """
    Set up a QA chain with the specified vectorstore.
    
    Args:
        vectorstore: The vector store to use
        output_tokens: Max output tokens
        temperature: Temperature for generation
        balance_multimodal: Whether to use balanced multimodal retrieval
        modality_distribution: Distribution of document types if balance_multimodal is True
        
    Returns:
        The configured QA chain
    """
    llm = CustomOllama(
        model="deepseek-r1:70b",
        base_url="http://localhost:11434",
        temperature=temperature,
        max_tokens=output_tokens
    )
    
    # Use balanced retriever if requested
    if balance_multimodal and modality_distribution:
        retriever = create_balanced_retriever(
            vectorstore=vectorstore, 
            initial_k=10, 
            final_k=5,
            modality_distribution=modality_distribution
        )
        logger.info("Using balanced multimodal retriever")
    else:
        # Use our standard RerankingRetriever
        retriever = RerankingRetriever(vectorstore=vectorstore, initial_k=10, final_k=5)
        logger.info("Using standard reranking retriever")
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    logger.info("QA chain setup complete.")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

def setup_qa_chain_with_retriever(retriever, output_tokens: int, temperature: float):
    """
    Set up a QA chain with a specific retriever.
    
    Args:
        retriever: The retriever to use
        output_tokens: Max output tokens
        temperature: Temperature for generation
        
    Returns:
        The configured QA chain
    """
    llm = CustomOllama(
        model="deepseek-r1:70b",
        base_url="http://localhost:11434",
        temperature=temperature,
        max_tokens=output_tokens
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    logger.info("QA chain setup complete with custom retriever.")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

def clear_conversation(qa_chain):
    """Clear conversation history from a QA chain."""
    if qa_chain and qa_chain.memory:
        qa_chain.memory.clear()
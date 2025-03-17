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
    import re  # Make sure re is imported for the EnhancedQAChain class
    print("DEBUG: Setting up QA chain with retriever")
    
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
    
    # Create standard QA chain
    standard_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    # Wrap with enhanced chain for image processing
    print("DEBUG: Creating EnhancedQAChain")
    enhanced_chain = EnhancedQAChain(standard_chain)
    
    print("DEBUG: Enhanced QA chain created")
    logger.info("QA chain setup complete with custom retriever and image enhancement.")
    return enhanced_chain

def clear_conversation(qa_chain):
    """Clear conversation history from a QA chain."""
    if qa_chain and qa_chain.memory:
        qa_chain.memory.clear()

class EnhancedQAChain:
    """Simple wrapper around QA chain to add image recaptioning"""
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.memory = qa_chain.memory
        # Get the LLM directly during initialization to avoid attribute errors
        if hasattr(qa_chain, 'llm'):
            self.llm = qa_chain.llm
        else:
            # Create a new LLM instance that matches what's used in the main code
            self.llm = CustomOllama(
                model="deepseek-r1:70b",
                base_url="http://localhost:11434",
                temperature=0.6,
                max_tokens=8192
            )
    
    def invoke(self, input_dict):
        """Process query and enhance with image analysis if needed"""
        import re
        from config import BASE_DIR
        import os
        import glob
        from image_processing import recaption_image_for_query
        
        query = input_dict["question"]
        
        print(f"DEBUG: EnhancedQAChain received query: {query}")
        
        # First use standard QA chain
        result = self.qa_chain.invoke(input_dict)
        
        # Have the LLM determine if visual analysis would be helpful
        retriever = self.qa_chain.retriever
        docs = retriever.get_relevant_documents(query)
        print(f"DEBUG: Retrieved {len(docs)} documents")
        
        # Check if any of the retrieved documents contain image content
        has_image_content = False
        for i, doc in enumerate(docs):
            print(f"DEBUG: Document {i+1} metadata: {doc.metadata}")
            content_snippet = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"DEBUG: Document {i+1} content snippet: {content_snippet}")
            
            if ('type' in doc.metadata and doc.metadata['type'] == 'image') or \
               ('file_path' in doc.metadata) or \
               ('[IMAGE CAPTION' in doc.page_content):
                has_image_content = True
                print(f"DEBUG: Document {i+1} contains image content")
                break
        
        if not has_image_content:
            print("DEBUG: No image content found in retrieved documents")
            return result
        
        # Let's use a simpler approach than asking the LLM
        # If there's image content in the retrieved docs, we'll just assume visual analysis is helpful
        print("DEBUG: Image content found, proceeding with visual analysis")
        
        try:
            # Look for image paths in metadata or references in text
            images_to_analyze = []
            
            # Check for file_path in metadata
            for i, doc in enumerate(docs):
                if 'file_path' in doc.metadata:
                    images_to_analyze.append(doc.metadata['file_path'])
                    print(f"DEBUG: Found file_path in metadata: {doc.metadata['file_path']}")
                
                # Also check page content for image captions that include file paths
                for line in doc.page_content.split('\n'):
                    if 'file_path' in line and '.jpg' in line:
                        print(f"DEBUG: Found potential file_path in content: {line}")
                        path_match = re.search(r'file_path: ([^\s,]+\.jpg)', line)
                        if path_match:
                            path = path_match.group(1)
                            print(f"DEBUG: Extracted path: {path}, exists: {os.path.exists(path)}")
                            if os.path.exists(path):
                                images_to_analyze.append(path)
            
            # If we didn't find paths in doc metadata, try page numbers
            # Inside the EnhancedQAChain.invoke method:
            if not images_to_analyze:
                print("DEBUG: No image paths found in metadata, trying page numbers")
                for doc in docs:
                    if 'source' in doc.metadata and 'page' in doc.metadata:
                        source_path = doc.metadata['source']
                        page_num = doc.metadata['page']
                        
                        # Try different ways of extracting document_id
                        doc_id = os.path.basename(source_path).replace('.pdf', '')
                        print(f"DEBUG: Doc ID from filename: {doc_id}")
                        
                        # Try extracting from full path
                        path_parts = source_path.split('/')
                        for i in range(len(path_parts)):
                            if i >= 2 and path_parts[i-2] == 'uploads':
                                potential_id = path_parts[i]
                                print(f"DEBUG: Potential doc ID from path: {potential_id}")
                        
                        # Try multiple patterns with wildcards
# Add to the patterns list:
                        patterns = [
                            # JPG patterns
                            os.path.join(BASE_DIR, "extracted_images", doc_id, f"page{page_num}_*.jpg"),
                            # PNG patterns
                            os.path.join(BASE_DIR, "extracted_images", doc_id, f"page{page_num}_*.png"),
                            # Other formats
                            os.path.join(BASE_DIR, "extracted_images", doc_id, f"page{page_num}_*.jpeg"),
                            os.path.join(BASE_DIR, "extracted_images", doc_id, f"page{page_num}_*.tiff"),
                            os.path.join(BASE_DIR, "extracted_images", doc_id, f"page{page_num}_*.gif")
                        ]
                                                
                        for pattern in patterns:
                            print(f"DEBUG: Looking for images with pattern: {pattern}")
                            matching_images = glob.glob(pattern)
                            print(f"DEBUG: Found {len(matching_images)} matching images: {matching_images}")
                            if matching_images:
                                images_to_analyze.extend(matching_images)
            
            print(f"DEBUG: Images to analyze: {images_to_analyze}")
            
            # Recaption found images
            if images_to_analyze:
                enhanced_captions = []
                for img_path in images_to_analyze[:2]:  # Limit to 2 images
                    if os.path.exists(img_path):
                        print(f"DEBUG: Generating query-specific caption for: {img_path}")
                        caption = recaption_image_for_query(img_path, query)
                        print(f"DEBUG: Generated caption: {caption[:100]}...")  # First 100 chars
                        enhanced_captions.append(caption)
                    else:
                        print(f"DEBUG: Image file does not exist: {img_path}")
                
                if enhanced_captions:
                    print(f"DEBUG: Generated {len(enhanced_captions)} enhanced captions")
                    # Enhance the answer with new captions
                    enhanced_answer = f"{result['answer']}\n\nAdditional image analysis:\n"
                    for i, caption in enumerate(enhanced_captions, 1):
                        enhanced_answer += f"\nImage {i}: {caption}\n"
                    
                    result['answer'] = enhanced_answer
                else:
                    print("DEBUG: No enhanced captions were generated")
            else:
                print("DEBUG: No images found to analyze")
            
            return result
            
        except Exception as e:
            # If anything fails, return the original result
            print(f"DEBUG: Error in image enhancement: {e}")
            import traceback
            traceback.print_exc()
            return result
# vector_store.py
import os
import streamlit as st
import logging
import json
import itertools
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_CACHE_DIR

logger = logging.getLogger(__name__)

def build_vector_store(docs, embedding_model_name: str):
    """
    Build a vector store from a list of documents.
    
    Args:
        docs: List of Document objects
        embedding_model_name: Name of the embedding model to use
        
    Returns:
        The created vector store
    """
    st.write(f"Building vector store for {len(docs)} document chunks...")
    
    # Count document types for progress reporting
    doc_types = {}
    for doc in docs:
        doc_type = doc.metadata.get("type", "text")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    type_info = ", ".join([f"{count} {doc_type}" for doc_type, count in doc_types.items()])
    st.write(f"Document types: {type_info}")
    
    # Initialize embeddings model - GPU device changed to 0
    model_kwargs = {'device': 'cuda:0'}
    if embedding_model_name == "nomic-ai/nomic-embed-text-v1":
        model_kwargs['trust_remote_code'] = True
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )
    
    with st.spinner("Creating FAISS vector store..."):
        vectorstore = FAISS.from_documents(docs, embeddings)
    
    st.success("Vector store created successfully!")
    logger.info(f"Vector store built successfully with {len(docs)} documents: {type_info}")
    return vectorstore

def load_cached_vectorstore(cache_path: str, embedding_model_name: str):
    """
    Load a vector store from a cache directory.
    
    Args:
        cache_path: Path to the cache directory
        embedding_model_name: Name of the embedding model to use
        
    Returns:
        The loaded vector store, or None if it couldn't be loaded
    """
    if not os.path.exists(cache_path):
        return None
    try:
        st.write(f"Loading cached vector store from: {cache_path}")
        # GPU device changed to 0
        model_kwargs = {'device': 'cuda:0'}
        if embedding_model_name == "nomic-ai/nomic-embed-text-v1":
            model_kwargs['trust_remote_code'] = True
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs
        )
        vectorstore = FAISS.load_local(
            folder_path=cache_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Try to load metadata
        metadata_path = os.path.join(cache_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            st.write(f"Loaded vector store with {metadata.get('total_docs', 'unknown')} documents")
            if "doc_types" in metadata:
                type_info = ", ".join([f"{count} {doc_type}" for doc_type, count in metadata["doc_types"].items()])
                st.write(f"Document types: {type_info}")
        
        logger.info("Cached vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        st.warning(f"Failed to load cached vector store, regenerating embeddings. Exception: {e}")
        logger.exception("Error loading cached vector store, regenerating embeddings.")
        return None

def save_vectorstore(vectorstore, cache_path: str, docs=None):
    """
    Save a vector store to a cache directory.
    
    Args:
        vectorstore: The vector store to save
        cache_path: Path to the cache directory
        docs: Optional list of Document objects for metadata
    """
    os.makedirs(cache_path, exist_ok=True)
    vectorstore.save_local(folder_path=cache_path)
    
    # Save metadata if documents are provided
    if docs:
        doc_types = {}
        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        metadata = {
            "total_docs": len(docs),
            "doc_types": doc_types
        }
        
        with open(os.path.join(cache_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    logging.getLogger(__name__).info("Vector store saved to cache.")

def create_multimodal_vectorstore(text_docs, table_docs, image_docs, embedding_model_name):
    """
    Create a unified vector store containing text, table, and image documents.
    
    Args:
        text_docs: List of Document objects containing text chunks
        table_docs: List of Document objects containing table data
        image_docs: List of Document objects containing image captions
        embedding_model_name: The embedding model to use
        
    Returns:
        The created vector store
    """
    # Combine all documents
    all_docs = list(itertools.chain(text_docs, table_docs, image_docs))
    
    # Create the vector store
    vectorstore = build_vector_store(all_docs, embedding_model_name)
    
    # Log information about the created store
    logger.info(f"Created multimodal vector store with:")
    logger.info(f"- {len(text_docs)} text documents")
    logger.info(f"- {len(table_docs)} table documents")
    logger.info(f"- {len(image_docs)} image documents")
    
    return vectorstore, all_docs
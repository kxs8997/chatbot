# main.py
import os
import time
import re
import streamlit as st
import logging
import nltk

# Ensure NLTK's sentence tokenizer is downloaded.
nltk.download('punkt')
nltk.download('punkt_tab')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import UPLOADS_DIR, DATASOURCES, DEFAULT_CHUNK_SIZE, DEFAULT_OUTPUT_TOKENS, VECTOR_CACHE_DIR, DEFAULT_TEMPERATURE, BASE_DIR
from auth import setup_authentication
from ftp_utils import test_ftp_connection, upload_to_ftp
from document_processing import parse_files, get_supported_extensions, parse_mediawiki_dump, pages_to_documents
from document_processing import extract_tables_from_pdf, enhanced_document_chunking
from vector_store import build_vector_store, load_cached_vectorstore, save_vectorstore
from qa_chain import clear_conversation, CustomOllama, RerankingRetriever
from qa_chain import create_balanced_retriever, setup_qa_chain_with_retriever
from utils import format_answer, extract_thoughts
from image_processing import extract_images_and_captions
from langchain.docstore.document import Document
import itertools

st.set_page_config(page_title="Mabl-GPT", layout="wide")

# --- NLTK-based semantic splitting functions ---
from nltk.tokenize import sent_tokenize

def semantic_split_nltk(text, max_length=1024, overlap=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def semantic_text_splitter_nltk(docs, max_length=1024, overlap=200):
    from langchain.docstore.document import Document
    new_docs = []
    for doc in docs:
        # Skip non-text documents
        if doc.metadata.get("type") != "text" and doc.metadata.get("type") != "code":
            new_docs.append(doc)
            continue
            
        for chunk in semantic_split_nltk(doc.page_content, max_length, overlap):
            new_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return new_docs
# --- End of NLTK-based semantic splitting functions ---

def render_llm_output(text: str):
    """
    Render LLM output that may contain LaTeX.
    Converts block and inline math to $$...$$ and renders alternately as markdown and LaTeX.
    """
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\s*$', '', text, flags=re.MULTILINE)
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        if re.search(r'^\s*[\(\-\*\u2022]', line) and re.search(r'\\(sin|cos|frac|text|sqrt)', line):
            if "$$" not in line:
                line = "$$" + line.strip("() \n") + "$$"
        processed_lines.append(line)
    processed_text = "\n".join(processed_lines)
    if "$$" in processed_text:
        parts = processed_text.split("$$")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                st.markdown(part)
            else:
                st.latex(part.strip())
    else:
        st.markdown(processed_text)

def setup_direct_chat_chain(output_tokens, temperature):
    """
    Create and return a direct chat chain.
    """
    from langchain.memory import ConversationBufferMemory
    llm = CustomOllama(
        model="deepseek-r1:70b",
        base_url="http://localhost:11434",
        temperature=temperature,
        max_tokens=output_tokens
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def invoke(input_dict):
        query = input_dict["question"]
        history = ""
        for msg in memory.chat_memory.messages:
            if msg.__class__.__name__ == "HumanMessage":
                history += "User: " + msg.content + "\n"
            elif msg.__class__.__name__ == "AIMessage":
                history += "Assistant: " + msg.content + "\n"
        prompt = history + "User: " + query + "\nAssistant: "
        response = llm(prompt)
        from langchain.schema import HumanMessage, AIMessage
        memory.chat_memory.add_message(HumanMessage(content=query))
        memory.chat_memory.add_message(AIMessage(content=response))
        return {"answer": response}
    
    class DirectChatChain:
        def __init__(self, invoke_fn, memory):
            self.invoke = invoke_fn
            self.memory = memory
    return DirectChatChain(invoke, memory)

def handle_file_uploads(ftp_available: bool, container):
    """
    Render the file uploader inside the container.
    """
    container.header("Upload Documents", help="Upload PDF or code files to be processed.")
    supported_extensions = get_supported_extensions()
    uploaded_files = container.file_uploader(
        "Upload Files",
        type=supported_extensions,
        accept_multiple_files=True,
        key="multi_file_uploader",
        help="Select files to upload (PDFs and code files supported)."
    )
    if uploaded_files and container.button("Process Uploads", key="process_uploads_button", help="Process uploads and add them to your collection."):
        if not ftp_available:
            container.error("Cannot upload: FTP server not available.")
            return
        processed_files = []
        with st.spinner("Processing uploads..."):
            for uploaded_file in uploaded_files:
                try:
                    bytes_data = uploaded_file.getvalue()
                    remote_path = upload_to_ftp(bytes_data, uploaded_file.name)
                    if remote_path:
                        processed_files.append(remote_path)
                        container.success(f"Uploaded: {uploaded_file.name}")
                except Exception as e:
                    container.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    logger.exception(f"Upload failed for {uploaded_file.name}")
        if processed_files:
            st.session_state.current_documents = processed_files
            st.session_state.upload_version = time.time()
            container.success(f"Successfully processed {len(processed_files)} file(s)")

def reload_data(
    selected_source, 
    chunk_size, 
    chunk_overlap, 
    embedding_model_name, 
    output_tokens, 
    temperature, 
    force_regenerate, 
    use_semantic_splitter=False,
    use_enhanced_captions=True,  # New parameter
    process_tables_separately=True,    # New parameter
    balance_multimodal=True           # New parameter
):
    """
    Enhanced reload_data function with multimodal support.
    
    Args:
        selected_source: The data source to load
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        embedding_model_name: Name of embedding model to use
        output_tokens: Max output tokens for the model
        temperature: Temperature for generation
        force_regenerate: Whether to regenerate embeddings
        use_semantic_splitter: Whether to use semantic splitting
        use_enhanced_captions: Whether to use enhanced image captioning
        process_tables_separately: Whether to process tables as separate documents
        balance_multimodal: Whether to balance multimodal content in retrieval
    """
    username = st.session_state.username
    st.write("Loading data...")
    # Use a unique QA chain key per source
    qa_chain_key = f"qa_chain_{selected_source}_{username}"
    try:
        if selected_source == "USER_UPLOADS":
            if st.session_state.get("current_documents"):
                DATASOURCES["USER_UPLOADS"]["filenames"] = st.session_state.current_documents
                upload_version = st.session_state.get("upload_version", 0)
            else:
                st.error("Please upload and process documents first.")
                return
        else:
            upload_version = 0

        # For non-user-upload sources, temporarily disable filtering in parse_files.
        original_current_documents = st.session_state.get("current_documents", None)
        if selected_source != "USER_UPLOADS":
            st.session_state.current_documents = None

        # Track different document types separately
        text_docs = []
        table_docs = []
        image_docs = []
        
        # Parse documents based on data source type, handling each modality
        if DATASOURCES[selected_source]["type"] in ["pdf", "mixed"]:
            # Get regular document text
            docs = parse_files(
                DATASOURCES[selected_source]["filenames"],
                process_tables_separately=process_tables_separately,
                use_enhanced_captions=use_enhanced_captions
            )
            text_docs.extend(docs)
            
            # Process tables separately if requested
            if process_tables_separately:
                for filename in DATASOURCES[selected_source]["filenames"]:
                    if filename.lower().endswith('.pdf'):
                        with st.spinner(f"Extracting tables from {os.path.basename(filename)}..."):
                            tables = extract_tables_from_pdf(filename)
                            if tables:
                                st.success(f"Extracted {len(tables)} tables")
                                table_docs.extend(tables)
            
            # Process images with enhanced captioning if requested
            for filename in DATASOURCES[selected_source]["filenames"]:
                if filename.lower().endswith('.pdf'):
                    with st.spinner(f"Processing images in {os.path.basename(filename)}..."):
                        debug_mode = st.session_state.get("debug_captions", False)
                        page_captions = extract_images_and_captions(
                            filename, 
                            debug=debug_mode,
                            use_enhanced_captions=use_enhanced_captions
                        )
                        
                        # Convert image captions to documents
                        for page_num, images in page_captions.items():
                            for idx, img_info in enumerate(images):
                                caption = img_info.get("caption", "")
                                img_type = img_info.get("type", "unknown")
                                
                                if caption:
                                    doc = Document(
                                        page_content=caption,
                                        metadata={
                                            "source": filename,
                                            "page": page_num + 1,
                                            "image_index": idx,
                                            "type": "image",
                                            "image_type": img_type,
                                            "width": img_info.get("width", 0),
                                            "height": img_info.get("height", 0)
                                        }
                                    )
                                    image_docs.append(doc)
                        
                        if image_docs:
                            st.success(f"Processed {len(image_docs)} images with captions")
        else:
            st.error("Unsupported data source.")
            return

        # Restore the original current_documents value.
        if selected_source != "USER_UPLOADS":
            st.session_state.current_documents = original_current_documents

        # Split text documents into chunks - table and image docs are kept as is
        if use_semantic_splitter:
            text_doc_chunks = semantic_text_splitter_nltk(text_docs, max_length=chunk_size, overlap=chunk_overlap)
            st.success(f"Split {len(text_docs)} text documents into {len(text_doc_chunks)} chunks using semantic splitting")
        else:
            text_doc_chunks = enhanced_document_chunking(text_docs, max_length=chunk_size, overlap=chunk_overlap)
            st.success(f"Split {len(text_docs)} text documents into {len(text_doc_chunks)} chunks using enhanced document chunking")

        # Combine all document types for the vectorstore
        all_docs = text_doc_chunks + table_docs + image_docs
        
        if len(all_docs) == 0:
            st.error("No document chunks were created. Check your source files.")
            return
        
        logger.info(f"Prepared {len(text_doc_chunks)} text chunks, {len(table_docs)} tables, and {len(image_docs)} image captions.")

        # Determine cache path based on source - Removed WIKI and Argus Process Docs conditions
        if selected_source == "USER_UPLOADS":
            cache_path = os.path.join(VECTOR_CACHE_DIR, f"{selected_source}_vectorstore_{username}_{st.session_state.upload_version}")
        else:
            cache_path = os.path.join(VECTOR_CACHE_DIR, f"{selected_source}_vectorstore")

        # Load or build the vector store
        if not force_regenerate:
            vectorstore = load_cached_vectorstore(cache_path, embedding_model_name)
        else:
            vectorstore = None
            
        if vectorstore is None:
            with st.spinner("Building vector store..."):
                vectorstore = build_vector_store(all_docs, embedding_model_name)
                save_vectorstore(vectorstore, cache_path, all_docs)

        if not vectorstore:
            st.error("Vector store creation failed.")
            return
            
        # Create the retriever with multimodal balancing if requested
        if balance_multimodal and (len(table_docs) > 0 or len(image_docs) > 0):
            retriever = create_balanced_retriever(
                vectorstore=vectorstore,
                initial_k=10, 
                final_k=5,
                # Track document type distribution for balanced retrieval
                modality_distribution={
                    "text": len(text_doc_chunks),
                    "table": len(table_docs),
                    "image": len(image_docs)
                }
            )
            st.success("Created balanced multimodal retriever")
        else:
            # Use the standard retriever
            retriever = RerankingRetriever(
                vectorstore=vectorstore,
                initial_k=10,
                final_k=5
            )
            
        # Set up QA chain with the retriever
        st.write("Setting up QA chain...")
        st.session_state[qa_chain_key] = setup_qa_chain_with_retriever(
            retriever=retriever,
            output_tokens=output_tokens,
            temperature=temperature
        )
        
        st.success(f"Successfully processed {len(all_docs)} documents ({len(text_doc_chunks)} text, {len(table_docs)} tables, {len(image_docs)} images)!")
        logger.info("QA chain is ready.")

        # Cleanup: Delete temporary variables and clear GPU cache
        del text_docs, text_doc_chunks, table_docs, image_docs, all_docs
        import torch
        torch.cuda.empty_cache()

    except Exception as e:
        st.error(f"Error in reload_data: {str(e)}")
        logger.exception("Error in reload_data.")

def main():
    if not setup_authentication():
        return

    st.title("Mabl-GPT")
    st.warning("Mabl-GPT is highly experimental. Please verify all output before relying on it.")
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "extracted_images"), exist_ok=True)

    # ───── Sidebar Controls ─────
    with st.sidebar:
        st.markdown(
            "<h3>How to clear the conversation <span style='font-size:18px; cursor:help;' title='Clear Conversation resets the session; Clear Memory clears only the chat history while keeping your documents intact.'>❓</span></h3>",
            unsafe_allow_html=True,
        )
        # Clear controls grouped in an expander.
        with st.expander("Clear Controls", expanded=True):
            st.info("Clear Conversation resets the entire session; Clear Memory clears only the chat history while retaining your documents.")
            col1, col2 = st.columns(2)
            if col1.button("Clear Conversation", key="clear_conv", help="Reset the entire chat session."):
                qa_chain_key = f"qa_chain_{st.session_state.username}"
                if qa_chain_key in st.session_state:
                    del st.session_state[qa_chain_key]
                if "direct_chat_chain" in st.session_state:
                    del st.session_state["direct_chat_chain"]
                st.success("Conversation cleared.")
            if col2.button("Clear Memory", key="clear_mem", help="Clear only the conversation history."):
                qa_chain_key = f"qa_chain_{st.session_state.username}"
                if qa_chain_key in st.session_state and st.session_state[qa_chain_key]:
                    clear_conversation(st.session_state[qa_chain_key])
                if st.session_state.get("direct_chat_chain"):
                    st.session_state.direct_chat_chain.memory.clear()
                st.success("Memory cleared.")

        interface_mode = st.radio(
            "Interface Mode",
            ("Simple Mode", "Advanced Mode"),
            index=0,
            key="interface_mode_radio",
            help="Simple: minimal controls; Advanced: additional settings."
        )
        chat_mode = st.radio(
            "Chat Mode",
            ("Direct Chat", "Document Chat"),
            index=0,
            key="chat_mode_radio",
            help="Direct Chat interacts directly with the model; Document Chat uses your documents."
        )

        # Chat-mode specific options.
        if chat_mode == "Document Chat":
            with st.expander("Document Chat Options", expanded=True):
                embedding_model_name = st.selectbox(
                    "Select Embedding Model",
                    [
                        "nomic-ai/nomic-embed-text-v1",
                        "all-MiniLM-L6-v2",
                        "sentence-transformers/all-mpnet-base-v2",
                        "sentence-transformers/multi-qa-mpnet-base-cos-v1"
                    ],
                    key="embedding_model",
                    help="Select the model for document embeddings."
                )
                force_regenerate = st.checkbox(
                    "Force Regenerate Embeddings",
                    value=False,
                    key="force_regen",
                    help="Rebuild embeddings even if cached."
                )
                use_semantic_splitter = st.checkbox(
                    "Use NLTK-based Semantic Splitting",
                    value=True,
                    key="semantic_splitter",
                    help="Use NLTK to split documents into chunks."
                )
                
                # New multimodal RAG controls
                use_enhanced_captions = st.checkbox(
                    "Use Enhanced Image Captions",
                    value=True,
                    key="enhanced_captions",
                    help="Use specialized prompts for different image types."
                )
                
                process_tables_separately = st.checkbox(
                    "Process Tables Separately",
                    value=True,
                    key="tables_separately",
                    help="Extract tables as separate documents."
                )
                
                balance_multimodal = st.checkbox(
                    "Balance Multimodal Results",
                    value=True,
                    key="balance_multimodal",
                    help="Balance retrieval across text, tables, and images."
                )
                
                selected_source = st.radio(
                    "Select Data Source",
                    list(DATASOURCES.keys()),
                    key="data_source_radio",
                    help="Choose which documents to use."
                )
                if selected_source == "USER_UPLOADS":
                    handle_file_uploads(test_ftp_connection(), container=st)
                if interface_mode == "Advanced Mode":
                    chunk_size = st.number_input(
                        "Chunk Size",
                        value=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                        key="chunk_size_input",
                        help="Max characters per document chunk."
                    )
                    chunk_overlap = st.number_input(
                        "Chunk Overlap",
                        value=st.session_state.get("chunk_overlap", 200),
                        key="chunk_overlap_input",
                        help="Overlap between chunks."
                    )
                    max_token_outputs = st.number_input(
                        "Max Token Outputs",
                        value=st.session_state.get("max_token_outputs", DEFAULT_OUTPUT_TOKENS),
                        key="max_token_outputs_input",
                        help="Max tokens for the model's response."
                    )
                    temperature = st.slider(
                        "Temperature",
                        0.0, 1.0,
                        value=st.session_state.get("temperature", DEFAULT_TEMPERATURE),
                        key="temperature_input",
                        help="Lower: more deterministic; higher: more creative."
                    )
                else:
                    chunk_size = DEFAULT_CHUNK_SIZE
                    chunk_overlap = 200
                    max_token_outputs = DEFAULT_OUTPUT_TOKENS
                    temperature = DEFAULT_TEMPERATURE

                if st.button("Load / Reload Data", key="reload_data_button", help="Process the selected documents and update the vector store."):
                    reload_data(
                        selected_source, 
                        chunk_size, 
                        chunk_overlap, 
                        embedding_model_name, 
                        max_token_outputs, 
                        temperature, 
                        force_regenerate, 
                        use_semantic_splitter,
                        use_enhanced_captions,
                        process_tables_separately,
                        balance_multimodal
                    )
        elif chat_mode == "Direct Chat":
            if interface_mode == "Advanced Mode":
                with st.expander("Direct Chat Options", expanded=True):
                    output_tokens = st.number_input(
                        "Direct Chat Max Token Outputs",
                        value=st.session_state.get("direct_max_token_outputs", DEFAULT_OUTPUT_TOKENS),
                        key="direct_max_token_outputs_input",
                        help="Max tokens for direct chat responses."
                    )
                    temperature_val = st.slider(
                        "Direct Chat Temperature",
                        0.0, 1.0,
                        value=st.session_state.get("direct_temperature", DEFAULT_TEMPERATURE),
                        key="direct_temperature_input",
                        help="Adjust the creativity for direct chat responses."
                    )
            else:
                output_tokens = DEFAULT_OUTPUT_TOKENS
                temperature_val = DEFAULT_TEMPERATURE

    # ───── End Sidebar Controls ─────

    # For Direct Chat, create the chain if not present.
    if chat_mode == "Direct Chat" and "direct_chat_chain" not in st.session_state:
        st.session_state.direct_chat_chain = setup_direct_chat_chain(output_tokens, temperature_val)

    # ───── Main Chat Interface ─────
    st.header("Chat with Mabl-GPT" if chat_mode == "Direct Chat" else "Chat with Your Documents")
    if chat_mode == "Document Chat":
        # Use a unique key for the QA chain per data source.
        qa_chain_key = f"qa_chain_{selected_source}_{st.session_state.username}"
        if qa_chain_key not in st.session_state or not st.session_state[qa_chain_key]:
            st.warning("Please load data before starting the chat.")
        else:
            for msg in st.session_state[qa_chain_key].memory.chat_memory.messages:
                if msg.__class__.__name__ == "HumanMessage":
                    st.chat_message("user").markdown(format_answer(msg.content))
                elif msg.__class__.__name__ == "AIMessage":
                    thoughts, main_content = extract_thoughts(msg.content)
                    with st.chat_message("assistant"):
                        if thoughts:
                            with st.expander("Show Thought Process", expanded=False):
                                for thought in thoughts:
                                    st.markdown(format_answer(thought))
                        render_llm_output(format_answer(main_content))
            user_input = st.chat_input("Your message")
            if user_input:
                st.chat_message("user").markdown(format_answer(user_input))
                with st.spinner("Processing..."):
                    result = st.session_state[qa_chain_key].invoke({"question": user_input})
                answer = result.get("answer", result)
                thoughts, main_answer = extract_thoughts(answer)
                with st.chat_message("assistant"):
                    if thoughts:
                        with st.expander("Show Thought Process", expanded=False):
                            for thought in thoughts:
                                st.markdown(format_answer(thought))
                    render_llm_output(format_answer(main_answer))
    else:  # Direct Chat
        if "direct_chat_chain" not in st.session_state:
            st.warning("Direct chat is not set up properly.")
        else:
            for msg in st.session_state.direct_chat_chain.memory.chat_memory.messages:
                if msg.__class__.__name__ == "HumanMessage":
                    st.chat_message("user").markdown(format_answer(msg.content))
                elif msg.__class__.__name__ == "AIMessage":
                    thoughts, main_content = extract_thoughts(msg.content)
                    with st.chat_message("assistant"):
                        if thoughts:
                            with st.expander("Show Thought Process", expanded=False):
                                for thought in thoughts:
                                    st.markdown(format_answer(thought))
                        render_llm_output(format_answer(main_content))
            user_input = st.chat_input("Your message")
            if user_input:
                st.chat_message("user").markdown(format_answer(user_input))
                with st.spinner("Processing..."):
                    result = st.session_state.direct_chat_chain.invoke({"question": user_input})
                answer = result.get("answer", result)
                thoughts, main_answer = extract_thoughts(answer)
                with st.chat_message("assistant"):
                    if thoughts:
                        with st.expander("Show Thought Process", expanded=False):
                            for thought in thoughts:
                                st.markdown(format_answer(thought))
                    render_llm_output(format_answer(main_answer))
    # ───── End Main Chat Interface ─────

if __name__ == "__main__":
    main()
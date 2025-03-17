# D3 RAG Chat Bot

This project is a Retrieval-Augmented Generation (RAG) chatbot that enables users to query document sources (such as a MediaWiki XML dump or PDFs with images) using FAISS for vector search. Powered by Ollama and built with Streamlit, the chatbot supports both retrieval-based Document Chat and direct conversation with an LLM, all while maintaining a private session for each user.

## Features
- **Document Processing:**  
  Parses MediaWiki XML dumps and PDFs (with image captioning via Moondream) and splits them into manageable chunks.
- **Vector Store:**  
  Builds and caches a FAISS vector store for efficient similarity search. Common sources share a vector store, while user uploads are kept private.
- **Chat Modes:**  
  - **Direct Chat:** Engage directly with the LLM without document retrieval.  
  - **Document Chat:** Query pre-processed documents using a conversational retrieval chain.
- **Sidebar Customization:**  
  - **Interface Mode:** Choose between a simple interface (default settings) and an advanced interface (exposes fine-tuning parameters).  
  - **Chat Mode Selection:** Switch between Direct Chat and Document Chat.  
  - **Advanced Settings:**  
    - *For Document Chat:* Adjust parameters such as chunk size, chunk overlap, maximum token outputs, and temperature.  
    - *For Direct Chat:* Configure only maximum token outputs and temperature.
- **File Upload & Data Source Options:**  
  Upload your own documents (USER_UPLOADS) via FTP, or select pre-configured sources (e.g., Argus Process Docs, WIKI).
- **Conversational Memory:**  
  Maintains a conversation buffer for context-aware responses.
- **Caching:**  
  Caches processed documents and vector stores to avoid repeating expensive operations like Moondream image captioning.

## Prerequisites
- Python 3.8+
- A virtual environment (recommended)
- Required dependencies (see setup instructions)

## Setup

### 1. Create and Activate a Virtual Environment
For example, on Linux:
```bash
python3 -m venv d3_rag_env
source d3_rag_env/bin/activate
```

### 2. Install Dependencies
Install the necessary Python libraries:
```bash
pip install streamlit langchain langchain-community faiss-cpu ollama xml.etree.ElementTree
```

### 3. Install Ollama
Ollama must be installed separately. To install it:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
For further details, refer to [Ollama’s official site](https://ollama.com).

### 4. Prepare Your Data
Ensure you have either a MediaWiki XML dump (e.g., `D3Wiki.xml`) or PDFs (with images) for document processing.

## Running the Application

### 1. Start Ollama
Pull the required Ollama model before launching:
```bash
ollama pull phi4
```

### 2. Run the Streamlit Application
Launch the chatbot by running:
```bash
streamlit run wiki_rag.py
```

### 3. Interact with the Chatbot
- **For Document Chat:**  
  - Use the sidebar to select your data source and upload files if necessary.  
  - Click "Load / Reload Data" to process the documents and build the vector store.  
  - Start chatting by entering your query.
- **For Direct Chat:**  
  - Simply enter your message to interact directly with the LLM.

## Sidebar Controls and Their Purpose

### Interface Mode
- **Options:** Simple Mode, Advanced Mode  
- **Purpose:**  
  - **Simple Mode:** Uses default settings with a minimal interface for ease of use.  
  - **Advanced Mode:** Exposes additional configuration options to fine-tune document processing and LLM responses.

### Chat Mode
- **Options:** Direct Chat, Document Chat  
- **Purpose:**  
  - **Direct Chat:** Converse directly with the LLM without referencing documents.  
  - **Document Chat:** Query pre-processed documents using a retrieval-augmented conversation chain.

### Advanced Settings (Visible Only in Advanced Mode)
- **For Document Chat:**
  - **Chunk Size:** Sets the maximum size of text segments when splitting documents.
  - **Chunk Overlap:** Determines how much overlap exists between consecutive chunks to maintain context.
  - **Max Token Outputs:** Limits the maximum number of tokens the LLM can generate.
  - **Temperature:** Adjusts the randomness of the LLM's output (lower values yield more deterministic responses).
- **For Direct Chat:**
  - **Max Token Outputs:** Configures the response length of the LLM.
  - **Temperature:** Controls the variability of the LLM's responses.

### File Upload and Data Source Options (Document Chat)
- **Data Source Selection:**  
  Choose from pre-configured sources (e.g., Argus Process Docs, WIKI) or upload your own documents (USER_UPLOADS).  
- **File Upload Widget:**  
  Allows users to upload files via FTP for private processing.
- **Embedding Model:**  
  Select the embedding model used to generate document embeddings.
- **Force Regenerate Embeddings:**  
  Option to bypass cached vector stores and rebuild embeddings.
- **Use NLTK-based Semantic Splitting:**  
  Toggle advanced text splitting for better document segmentation.

### Clear Conversation
- **Button:** "Clear Conversation"  
- **Purpose:**  
  Resets the current conversation and clears the session's chat history.

### FTP Connection Indicator
- **Purpose:**  
  Displays the status of the FTP server to ensure file uploads are functioning properly.

## Troubleshooting
- **File Paths:**  
  Verify the paths to your MediaWiki XML dump or PDFs if they are not found.
- **Ollama:**  
  Ensure Ollama is running and the required models are pulled.
- **Dependencies:**  
  Reinstall missing dependencies with `pip install -r requirements.txt`.
- **Caching:**  
  Use the "Force Regenerate Embeddings" option if you suspect the cached data is outdated.

## Future Improvements
- Support for additional LLMs and embedding models.
- Optimization for processing large document dumps.
- Enhanced UI for improved conversation management and real-time feedback.

---

Enjoy using the D3 RAG Chat Bot, and feel free to contribute or suggest improvements!


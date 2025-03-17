# document_processing.py
import os
import json
import hashlib
import logging
import streamlit as st
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
import pdfplumber  # Added for table extraction
from langchain.docstore.document import Document
from image_processing import extract_images_and_captions
from config import BASE_DIR, UPLOADS_DIR, CODE_FILE_EXTENSIONS

logger = logging.getLogger(__name__)

def get_supported_extensions():
    """Return list of supported file extensions."""
    extensions = list(CODE_FILE_EXTENSIONS.keys())
    extensions.append('.pdf')
    return extensions

def detect_file_type(file_content: bytes) -> str:
    """Detect MIME type using python-magic."""
    import magic
    mime = magic.Magic(mime=True)
    return mime.from_buffer(file_content)

def is_supported_file(mime_type: str, filename: str) -> bool:
    """Check if file is supported based on its extension and MIME type."""
    ext = os.path.splitext(filename)[1].lower()
    if mime_type == 'application/pdf':
        return True
    supported_extensions = get_supported_extensions()
    if ext in supported_extensions:
        return True
    if mime_type.startswith("text/"):
        return True
    return False

def parse_mediawiki_dump(xml_path: str) -> list:
    """
    Parse a MediaWiki XML dump and return a list of pages as dictionaries.
    Each dictionary contains the title and text of the page.
    
    Args:
        xml_path: Path to the MediaWiki XML dump file
        
    Returns:
        List of dictionaries with title and text keys
    """
    pages = []
    try:
        # Convert to absolute path if it's not already
        if not os.path.isabs(xml_path):
            for search_dir in [os.getcwd(), BASE_DIR, UPLOADS_DIR]:
                potential_path = os.path.join(search_dir, xml_path)
                if os.path.exists(potential_path):
                    xml_path = potential_path
                    logger.info(f"Resolved wiki XML path to: {xml_path}")
                    break
                    
        # Check if the file exists before trying to open it
        if not os.path.exists(xml_path):
            logger.error(f"Wiki XML file not found: {xml_path}")
            return []
            
        context = ET.iterparse(xml_path, events=("end",))
        for event, elem in context:
            if elem.tag.endswith("page"):
                title_elem = elem.find("./{*}title")
                text_elem = elem.find(".//{*}text")
                title = title_elem.text if title_elem is not None else "No title"
                text = text_elem.text if text_elem is not None else ""
                pages.append({"title": title, "text": text})
                elem.clear()
        st.success(f"Successfully parsed {len(pages)} pages from wiki XML")
        logger.info(f"Parsed {len(pages)} pages from XML: {xml_path}")
        return pages
    except Exception as e:
        st.error(f"Error parsing wiki XML: {str(e)}")
        logger.exception(f"Error parsing wiki XML {xml_path}: {e}")
        return []



def pages_to_documents(pages: list) -> list:
    """
    Convert pages (as dictionaries) to a list of Document objects.
    
    Args:
        pages: List of dictionaries with title and text keys
        
    Returns:
        List of Document objects
    """
    logger.info("Converting pages to Document objects")
    return [
        Document(
            page_content=f"Title: {page['title']}\n\n{page['text']}",
            metadata={
                "title": page["title"],
                "type": "text"
            }
        )
        for page in pages
    ]


def parse_code_file(file_content: bytes, filename: str) -> list:
    """Parse a code file into a Document object."""
    try:
        text_content = file_content.decode('utf-8')
        doc = Document(
            page_content=text_content,
            metadata={
                "source": filename,
                "type": "code",
                "extension": os.path.splitext(filename)[1]
            }
        )
        return [doc]
    except UnicodeDecodeError as e:
        logger.error(f"Error decoding file {filename}: {e}")
        st.error(f"Error: {filename} appears to be binary or uses an unsupported encoding")
        return []

def get_pdf_cache_filename(pdf_path: str) -> str:
    """
    Generate a cache filename for a given PDF file.
    The cache file will be stored alongside the PDF file.
    """
    base, _ = os.path.splitext(pdf_path)
    return f"{base}.cache.json"

def convert_table_to_markdown(table):
    """
    Convert a table (list of rows) extracted by pdfplumber into a Markdown table.
    This enhanced version handles empty cells, header rows, and maintains alignment.
    
    Args:
        table: A list of rows, where each row is a list of cell values
        
    Returns:
        Markdown formatted table as string
    """
    if not table or len(table) == 0:
        return ""
    
    # Process header row
    header = table[0]
    header = [str(cell).strip() if cell is not None else "" for cell in header]
    
    # Calculate column widths for better formatting
    col_widths = [max(3, len(h)) for h in header]  # Min width of 3 chars
    for row in table[1:]:
        for i, cell in enumerate(row):
            cell_str = str(cell).strip() if cell is not None else ""
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell_str))
    
    # Build the markdown table
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---" for _ in header]) + " |\n"
    
    # Process data rows
    for row in table[1:]:
        processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
        md += "| " + " | ".join(processed_row) + " |\n"
    
    return md

def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from a PDF and convert them to Document objects.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects containing tables
    """
    table_docs = []
    try:
        # Convert to absolute path if it's not already
        if not os.path.isabs(pdf_path):
            for search_dir in [os.getcwd(), BASE_DIR, UPLOADS_DIR]:
                potential_path = os.path.join(search_dir, pdf_path)
                if os.path.exists(potential_path):
                    pdf_path = potential_path
                    logger.info(f"Resolved path to: {pdf_path}")
                    break
        
        # Check if the file exists before trying to open it
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []
        
        print(f"\n===== Extracting tables from: {os.path.basename(pdf_path)} =====")
            
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                if tables:
                    print(f"\n[Table extraction] Found {len(tables)} tables on page {page_num + 1}")
                
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    
                    # Convert the table to markdown format
                    md_table = convert_table_to_markdown(table)
                    
                    # Print the table to console
                    print(f"\n----- Table {table_idx + 1} on page {page_num + 1} -----")
                    print(md_table)
                    print("-" * 50)
                    
                    # Create a document for the table
                    doc = Document(
                        page_content=md_table,
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "type": "table"
                        }
                    )
                    table_docs.append(doc)
        
        if table_docs:
            print(f"\n[Table extraction] Extracted {len(table_docs)} tables total from {os.path.basename(pdf_path)}")
        else:
            print(f"\n[Table extraction] No tables found in {os.path.basename(pdf_path)}")
            
    except Exception as e:
        logger.error(f"Error extracting tables from {pdf_path}: {e}")
        print(f"\n[Table extraction ERROR] {str(e)}")
        
    return table_docs

def parse_pdf_with_images(pdf_path: str, debug: bool = False, process_tables_separately: bool = True, use_enhanced_captions: bool = True) -> list:
    """
    Parse a PDF file to extract text, image captions, and table data.
    Table data is extracted using pdfplumber and converted to Markdown.
    Caching is used only if the PDF is not a user upload (i.e. not in UPLOADS_DIR).
    Returns a list of Document objects.
    
    Args:
        pdf_path: Path to the PDF file
        debug: Whether to print debug information
        process_tables_separately: Whether to extract tables as separate documents
                                  (False means tables are included in text)
        use_enhanced_captions: Whether to use enhanced image captioning
        
    Returns:
        List of Document objects
    """
    # Check if this is a user upload by looking for timestamp pattern in path
    # User uploads have paths like 'uploads/username/20230517_120345/file.pdf'
    is_user_upload = '/uploads/' in pdf_path and any(part.isdigit() for part in pdf_path.split('/'))
    
    cache_filename = get_pdf_cache_filename(pdf_path)
    use_cache = not is_user_upload
    
    if debug:
        print(f"PDF Path: {pdf_path}")
        print(f"Is user upload: {is_user_upload}")
        print(f"Using cache: {use_cache}")

    if use_cache and os.path.exists(cache_filename):
        try:
            with open(cache_filename, "r", encoding="utf-8") as f:
                cached_docs = json.load(f)
            documents = [
                Document(page_content=doc_dict["page_content"], metadata=doc_dict["metadata"])
                for doc_dict in cached_docs
            ]
            logger.info(f"Loaded cached processed PDF from {cache_filename}")
            return documents
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_filename}: {e}")

    # Open the PDF with PyMuPDF and pdfplumber
    doc = fitz.open(pdf_path)
    pdf_plumber_doc = pdfplumber.open(pdf_path)
    captions_mapping = extract_images_and_captions(
        pdf_path, 
        debug=debug, 
        use_enhanced_captions=use_enhanced_captions
    )
    documents = []
    base_filename = os.path.basename(pdf_path)  # Get the file name for context
    
    for page_number in range(len(doc)):
        page = doc[page_number]
        page_text = page.get_text().strip()
        
        # Extract tables using pdfplumber for the same page.
        if not process_tables_separately:
            try:
                pdf_page = pdf_plumber_doc.pages[page_number]
                tables = pdf_page.extract_tables()
                if tables:
                    if debug:
                        print(f"Found {len(tables)} table(s) on page {page_number + 1}")
                    for idx, table in enumerate(tables):
                        md_table = convert_table_to_markdown(table)
                        # Updated table marker includes the file name for disambiguation.
                        table_marker = f"\n\n[TABLE - {base_filename} | Page {page_number + 1} | Table {idx + 1}]\n{md_table}\n"
                        page_text += table_marker
                else:
                    if debug:
                        print(f"No tables found on page {page_number + 1}")
            except Exception as e:
                if debug:
                    print(f"Error extracting tables on page {page_number+1}: {e}")
        
        # Append image captions if available.
        if page_number in captions_mapping:
            for idx, img_info in enumerate(captions_mapping[page_number]):
                caption = img_info.get("caption", "")
                width = img_info.get("width", "unknown")
                height = img_info.get("height", "unknown")
                img_type = img_info.get("type", "unknown")
                verbose_caption = (
                    f"\n\n[IMAGE CAPTION - Page {page_number + 1}, Image {idx + 1} "
                    f"(Type: {img_type}, Dimensions: {width}x{height})]\n{caption}"
                )
                page_text += verbose_caption
        
        if not page_text:
            page_text = f"[No text found on page {page_number + 1}]"
        
        document = Document(
            page_content=page_text,
            metadata={
                "page": page_number + 1, 
                "source": pdf_path,
                "type": "text"
            }
        )
        documents.append(document)
    
    pdf_plumber_doc.close()
    
    if use_cache:
        # Cache the processed documents for non-user-upload PDFs.
        try:
            with open(cache_filename, "w", encoding="utf-8") as f:
                json.dump([{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents], f)
            logger.info(f"Cached processed PDF to {cache_filename}")
        except Exception as e:
            logger.warning(f"Failed to cache processed PDF: {e}")
    
    return documents

def enhanced_document_chunking(docs, max_length=1024, overlap=200):
    """
    Enhanced document chunking that respects document structure and semantic boundaries.
    
    Args:
        docs: List of Document objects
        max_length: Maximum chunk size in characters
        overlap: Overlap size in characters
        
    Returns:
        List of chunked Document objects
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Improved separators list that better respects document structure
    separators = [
        "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",  # Headers
        "\n\n", "\n",                                           # Paragraphs
        ". ", "! ", "? ",                                       # Sentences
        ", ", "; ", ":",                                        # Phrases
        " ", ""                                                 # Words/chars
    ]
    
    # Create the enhanced text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=overlap,
        length_function=len,
        separators=separators
    )
    
    # Split the documents
    chunked_docs = text_splitter.split_documents(docs)
    
    # Ensure each chunk retains its document type
    for chunk in chunked_docs:
        if "type" not in chunk.metadata:
            chunk.metadata["type"] = "text"  # Default type
    
    return chunked_docs

def semantic_text_splitter_nltk(docs, max_length=1024, overlap=200):
    """
    Split documents into chunks respecting semantic boundaries using NLTK.
    
    Args:
        docs: List of Document objects
        max_length: Maximum length of each chunk
        overlap: Overlap size between chunks
        
    Returns:
        List of chunked Document objects
    """
    import nltk
    from nltk.tokenize import sent_tokenize
    from langchain.docstore.document import Document
    
    # Ensure NLTK's sentence tokenizer is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    def semantic_split(text, max_length=max_length, overlap=overlap):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max_length, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > max_length:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from the end of the previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    # Try to find a sentence boundary in the overlap
                    last_period = overlap_text.rfind('. ')
                    if last_period != -1:
                        current_chunk = overlap_text[last_period+2:]
                    else:
                        current_chunk = overlap_text
                else:
                    current_chunk = ""
            
            # Add the sentence to the current chunk
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    new_docs = []
    for doc in docs:
        # Skip non-text documents
        if doc.metadata.get("type") != "text" and doc.metadata.get("type") != "code":
            new_docs.append(doc)
            continue
            
        for chunk in semantic_split(doc.page_content, max_length, overlap):
            metadata = doc.metadata.copy()
            new_docs.append(Document(page_content=chunk, metadata=metadata))
    
    return new_docs

def parse_files(filenames: list, process_tables_separately: bool = True, use_enhanced_captions: bool = True) -> list:
    """
    Parse files given by their filenames into Document objects.
    
    Args:
        filenames: List of file paths
        process_tables_separately: Whether to process tables as separate documents
        use_enhanced_captions: Whether to use enhanced image captioning
        
    Returns:
        List of Document objects
    """
    documents = []
    st.write(f"Processing files: {filenames}")  # Debug statement
    if st.session_state.get("current_documents"):
        filenames = [f for f in filenames if f in st.session_state.current_documents]
    logger.info(f"Processing files: {filenames}")
    for filename in filenames:
        # Check if the path is absolute
        if not os.path.isabs(filename):
            full_path = None
            for search_dir in [os.getcwd(), BASE_DIR, UPLOADS_DIR]:
                # Try both direct path and walking the directory
                potential_path = os.path.join(search_dir, filename)
                if os.path.exists(potential_path):
                    full_path = potential_path
                    break
                    
                # If not found directly, walk the directory
                for root, dirs, files in os.walk(search_dir):
                    if os.path.basename(filename) in files:
                        full_path = os.path.join(root, os.path.basename(filename))
                        break
                if full_path:
                    break
            if full_path is None:
                st.error(f"Could not find {filename} in project directories")
                logger.error(f"File not found: {filename}")
                continue
        else:
            full_path = filename
            
        # Check if the file exists before continuing
        if not os.path.exists(full_path):
            st.error(f"File does not exist: {full_path}")
            logger.error(f"File does not exist: {full_path}")
            continue
            
        try:
            with open(full_path, 'rb') as f:
                file_content = f.read()
            mime_type = detect_file_type(file_content)
            logger.info(f"Detected MIME type for {filename}: {mime_type}")
            if mime_type == 'application/pdf':
                try:
                    debug = st.session_state.get("debug_captions", False)
                    docs = parse_pdf_with_images(
                        full_path, 
                        debug=debug, 
                        process_tables_separately=process_tables_separately,
                        use_enhanced_captions=use_enhanced_captions
                    )
                    st.success(f"Successfully loaded {len(docs)} pages (with image captions and tables)")
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Error processing PDF with images {full_path}: {str(e)}")
                    logger.exception(f"Error processing PDF with images {full_path}: {e}")
            elif is_supported_file(mime_type, filename):
                docs = parse_code_file(file_content, filename)
                if docs:
                    st.success(f"Successfully loaded code file: {filename}")
                    logger.info(f"Processed {filename} as code file")
                    documents.extend(docs)
            else:
                error_msg = f"Unsupported file type ({mime_type}) for {filename}"
                logger.warning(error_msg)
                st.warning(error_msg)
        except Exception as e:
            st.error(f"Failed to process file {full_path}: {str(e)}")
            logger.exception(f"Error processing file {full_path}: {e}")
    return documents
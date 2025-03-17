import os

# Base directories
BASE_DIR = os.path.join(os.getcwd(), "my_ftp_files")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_CACHE_DIR = os.path.join(os.getcwd(), "vector_cache")
IMAGE_STORAGE_DIR = os.path.join(BASE_DIR, "image_storage")

# Default configuration parameters
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_OUTPUT_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.6

# Data sources configuration - removed Argus_Process_Docs and WIKI
DATASOURCES = {
    "USER_UPLOADS": {"type": "mixed", "filenames": []}
}

# FTP configuration
FTP_CONFIG = {
    "host": "127.0.0.1",
    "user": "myuser",
    "password": "mypass",
    "upload_dir": "uploads",
    "port": 2121,
    "max_retries": 3,
    "retry_delay": 2
}

# Supported code file extensions mapping to MIME types
CODE_FILE_EXTENSIONS = {
    '.py': 'text/x-python',
    '.cpp': 'text/x-c++',
    '.c': 'text/x-c',
    '.h': 'text/x-c',
    '.hpp': 'text/x-c++',
    '.java': 'text/x-java',
    '.js': 'text/javascript',
    '.jsx': 'text/javascript',
    '.ts': 'text/x-typescript',
    '.tsx': 'text/x-typescript',
    '.go': 'text/x-go',
    '.rs': 'text/x-rust',
    '.txt': 'text/plain',
    '.md': 'text/plain',
    '.sql': 'text/plain',
    '.sh': 'text/plain',
    '.bat': 'text/plain'
}
# utils.py
import re
import time
import logging
import streamlit as st
from ftplib import error_perm

logger = logging.getLogger(__name__)

def format_answer(answer: str) -> str:
    """Clean and standardize the answer text."""
    answer = answer.strip()
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    return answer

def extract_thoughts(answer_text: str):
    """
    Extract any text between <think> and </think> tags.
    Returns a tuple: (list_of_thoughts, answer_without_thoughts).
    """
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    thoughts = pattern.findall(answer_text)
    answer_without_thoughts = pattern.sub('', answer_text).strip()
    return thoughts, answer_without_thoughts

def retry_ftp_operation(operation_func):
    """Decorator to retry FTP operations."""
    from config import FTP_CONFIG  # local import to avoid circular dependency issues
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(FTP_CONFIG["max_retries"]):
            try:
                result = operation_func(*args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt+1} failed in {operation_func.__name__}: {e}")
                if attempt < FTP_CONFIG["max_retries"] - 1:
                    time.sleep(FTP_CONFIG["retry_delay"])
                continue
        st.error(f"Failed after {FTP_CONFIG['max_retries']} attempts: {str(last_exception)}")
        logger.error(f"{operation_func.__name__} failed: {last_exception}")
        return None
    return wrapper

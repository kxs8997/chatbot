# ftp_utils.py
import os
import time
import logging
from ftplib import FTP, error_perm
from io import BytesIO
import streamlit as st
from datetime import datetime
from utils import retry_ftp_operation
from config import FTP_CONFIG

logger = logging.getLogger(__name__)

@retry_ftp_operation
def test_ftp_connection():
    try:
        with FTP() as ftp:
            ftp.connect(host=FTP_CONFIG["host"], port=FTP_CONFIG["port"])
            ftp.login(user=FTP_CONFIG["user"], passwd=FTP_CONFIG["password"])
            st.sidebar.success("✅ FTP Server Connected")
            logger.info("FTP connection successful.")
            return True
    except Exception as e:
        st.sidebar.error(f"❌ FTP Server Not Available: {str(e)}")
        logger.error(f"FTP connection error: {e}")
        return False

@retry_ftp_operation
def upload_to_ftp(file_data, filename):
    username = st.session_state.get("username", "anonymous")
    with FTP() as ftp:
        ftp.connect(host=FTP_CONFIG["host"], port=FTP_CONFIG["port"])
        ftp.login(user=FTP_CONFIG["user"], passwd=FTP_CONFIG["password"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = os.path.join('uploads', username, timestamp)
        # Create directories step by step
        for directory in ['uploads', username, timestamp]:
            try:
                ftp.mkd(directory)
            except error_perm as e:
                if not str(e).startswith('550'):
                    raise
            ftp.cwd(directory)
        bio = BytesIO(file_data)
        ftp.storbinary(f'STOR {filename}', bio)
        remote_full_path = os.path.join(upload_path, filename)
        logger.info(f"Uploaded file for user {username} to FTP: {remote_full_path}")
        return remote_full_path

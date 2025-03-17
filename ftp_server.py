import os
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import argparse
import logging

def setup_logger():
    """Configure logging for the FTP server"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('ftpserver')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def create_ftp_server(username, password, directory, host="127.0.0.1", port=2121):
    """
    Create and start an FTP server
    
    Parameters:
    - username: FTP user login name
    - password: FTP user password
    - directory: Root directory for FTP server
    - host: Host address to bind to (default: localhost)
    - port: Port to listen on (default: 2121)
    """
    logger = setup_logger()
    
    # Create main directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created main directory: {directory}")
    
    # Create uploads subdirectory
    upload_dir = os.path.join(directory, "uploads")
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        logger.info(f"Created uploads directory: {upload_dir}")
    
    # Set directory permissions (read/write for owner and group)
    os.chmod(directory, 0o775)
    os.chmod(upload_dir, 0o775)
    logger.info("Set directory permissions")

    # Initialize the authorizer
    authorizer = DummyAuthorizer()
    
    # Add user with full permissions
    # Permissions: "elradfmwMT"
    # e: change directory, l: list files, r: retrieve file
    # a: append data to file, d: delete file or directory
    # f: rename file or directory, m: create directory
    # w: store file, M: change file mode, T: change timestamp
    authorizer.add_user(username, password, directory, perm="elradfmwMT")
    
    # Create handler
    handler = FTPHandler
    handler.authorizer = authorizer
    
    # Set up passive ports (useful if behind firewall/NAT)
    handler.passive_ports = range(60000, 60100)
    
    # Create server
    server = FTPServer((host, port), handler)
    
    # Set maximum connections
    server.max_cons = 256
    server.max_cons_per_ip = 5
    
    logger.info(f"FTP server starting on {host}:{port}")
    logger.info(f"Root directory: {directory}")
    logger.info(f"Username: {username}")
    logger.info("Use Ctrl+C to stop the server")
    
    # Start server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down FTP server")
        server.close_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a simple FTP server")
    parser.add_argument("--username", default="user", help="FTP username")
    parser.add_argument("--password", default="password", help="FTP password")
    parser.add_argument("--directory", default="./ftp_root", help="Root directory for FTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=2121, help="Port to listen on")
    
    args = parser.parse_args()
    
    create_ftp_server(
        args.username,
        args.password,
        args.directory,
        args.host,
        args.port
    )
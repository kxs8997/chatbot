# image_processing.py
import io
import fitz  # PyMuPDF
import logging
import requests
import json
import time
import base64
from io import BytesIO
import subprocess
import os
import signal
import atexit
import threading
import socket
from PIL import Image
import numpy as np
from config import BASE_DIR, UPLOADS_DIR

logger = logging.getLogger(__name__)

# Global variables to track Ollama process
ollama_process = None
ollama_port = 11435  # Use a different port from DeepSeek
model_loaded = False
model_name = "granite3.2-vision"  # Can be replaced with "llava" or other models

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_ollama_server():
    """Start an Ollama server instance on the specified port"""
    global ollama_process
    
    # Check if port is already in use
    if is_port_in_use(ollama_port):
        logger.info(f"Port {ollama_port} is already in use. Assuming Ollama is running.")
        return True
    
    logger.info(f"Starting Ollama server on port {ollama_port}...")
    
    try:
        # Set the environment variable for the Ollama host
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{ollama_port}"
        # Updated to use GPU 1 instead of GPU 2
        env["CUDA_VISIBLE_DEVICES"] = "1"  # Changed from GPU 2 to GPU 1
        
        # Start Ollama as a subprocess
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Register function to shut down Ollama when the application exits
        atexit.register(stop_ollama_server)
        
        # Wait for server to start (look for specific output or try connecting)
        max_retries = 30
        for i in range(max_retries):
            if is_port_in_use(ollama_port):
                logger.info(f"Ollama server started successfully on port {ollama_port}")
                return True
            time.sleep(1)
            
        logger.error(f"Timed out waiting for Ollama server to start on port {ollama_port}")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama server: {e}")
        return False

def stop_ollama_server():
    """Stop the Ollama server"""
    global ollama_process
    if ollama_process:
        logger.info("Stopping Ollama server...")
        ollama_process.terminate()
        try:
            ollama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ollama_process.kill()
        ollama_process = None
        logger.info("Ollama server stopped")

def ensure_model_loaded():
    """Make sure the vision model is loaded"""
    global model_loaded
    
    if model_loaded:
        return True
        
    # Start the server if not already running
    if not start_ollama_server():
        logger.error("Failed to start Ollama server")
        return False
    
    # Check if model is already downloaded
    try:
        url = f"http://localhost:{ollama_port}/api/tags"
        response = requests.get(url)
        models = response.json().get("models", [])
        model_exists = any(m["name"] == model_name for m in models)
        
        # Pull the model if it doesn't exist
        if not model_exists:
            logger.info(f"Pulling {model_name} model...")
            pull_url = f"http://localhost:{ollama_port}/api/pull"
            pull_response = requests.post(pull_url, json={"name": model_name})
            
            if pull_response.status_code != 200:
                logger.error(f"Failed to pull {model_name} model: {pull_response.text}")
                return False
            logger.info(f"Successfully pulled {model_name} model")
        
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error ensuring model is loaded: {e}")
        return False

def fallback_caption(image):
    """Generate a simple caption with image dimensions when model fails"""
    width, height = image.size
    return f"Image of dimensions {width}x{height}"

def detect_image_type_with_granite(image, debug=False):
    """
    Use Granite vision model to detect the type of image
    
    Args:
        image: PIL Image object
        debug: Whether to print debug information
        
    Returns:
        String indicating image type ("text", "chart", "table", or "general")
    """
    try:
        # Ensure Ollama is running and model is loaded
        if not ensure_model_loaded():
            logger.warning("Could not ensure Ollama is running. Using fallback detection.")
            return "general"
        
        # Resize image for faster processing
        max_dimension = 512  # Smaller size for just type detection
        width, height = image.size
        
        # Only resize if the image is larger than the max dimension
        if width > max_dimension or height > max_dimension:
            scale_factor = max_dimension / max(width, height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            image_resized = image
            
        # Convert PIL Image to base64
        buffered = BytesIO()
        image_resized.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Create a simple prompt specifically for image type detection
        type_detection_prompt = """
        Look at this image and determine what type it is. Respond with EXACTLY ONE of these categories:
        - "chart" if the image is a chart, graph, plot, or any kind of data visualization
        - "general" for any other type of image
        
        Reply with just ONE word - the category name.
        """
        
        # Prepare the request for Ollama
        url = f"http://localhost:{ollama_port}/api/generate"
        
        # Create the request payload with lower tokens since we need just one word
        payload = {
            "model": model_name,
            "prompt": type_detection_prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 50  # Short response needed
            }
        }
        
        # Send the request to Ollama
        if debug:
            print("Sending image type detection request to Granite vision model...")
            
        response = requests.post(url, json=payload, timeout=15)  # Shorter timeout for type detection
        
        if response.status_code == 200:
            result = response.json()
            type_response = result.get("response", "").strip().lower()
            
            # Extract just the image type from the response (in case model outputs more text)
            if "chart" in type_response or "graph" in type_response:
                image_type = "chart"
            else:
                image_type = "general"
                
            if debug:
                print(f"Granite detected image type: {image_type}")
                print(f"Raw response: {type_response}")
                
            return image_type
        else:
            logger.error(f"Ollama API error in type detection: {response.status_code}")
            if debug:
                print(f"Type detection error: {response.status_code} - {response.text}")
            return "general"  # Default to general if detection fails
            
    except Exception as e:
        logger.error(f"Error in image type detection: {e}")
        if debug:
            print(f"Type detection error: {e}")
        return "general"  # Default to general on error

def get_prompt_for_image_type(image_type):
    """
    Get specialized prompt based on image type
    
    Args:
        image_type: String indicating image type
        
    Returns:
        Prompt string optimized for that image type
    """
    prompts = {
        "general": """
            Analyze this image thoroughly:
            1. Describe the main subjects and their positions
            2. Note any visible text and transcribe it exactly
            3. Describe the setting, background, and context
            4. Mention colors, styles, and notable features
            5. Identify any logos, brands, or recognizable elements
            Provide a comprehensive description that captures all important details.
        """,
        
        "chart": """
            This image contains a chart or graph. Analyze it by:
            1. Identifying the type of chart (bar, line, pie, scatter, etc.)
            2. Reading and reporting all axis labels, titles, and legends
            3. Extracting specific data points, values, and trends
            4. Stating maximum and minimum values
            5. Describing the overall pattern or conclusion
            Provide the analysis focused on the data visualization in a structured form that can be accessible programmatically.
        """
    }
    
    return prompts.get(image_type, prompts["general"]).strip()

def generate_caption_for_image(image, caption_length="normal", debug=False, image_type=None, use_enhanced_captions=True):
    """
    Generate a caption for an image using an advanced vision-language model
    
    Args:
        image: PIL Image object
        caption_length: Length of caption ("short", "normal", "detailed")
        debug: Whether to print debug information
        image_type: Override for image type detection
        use_enhanced_captions: Whether to use the enhanced captioning system
        
    Returns:
        Generated caption string
    """
    try:
        # Ensure Ollama is running and model is loaded
        if not ensure_model_loaded():
            logger.warning("Could not ensure Ollama is running. Using fallback caption.")
            return fallback_caption(image)
        
        # Detect image type with Granite if not specified and using enhanced captions
        if use_enhanced_captions:
            if image_type is None:
                image_type = detect_image_type_with_granite(image, debug=debug)
                if debug:
                    print(f"Granite detected image type: {image_type}")
            
            # Get the specialized prompt
            prompt = get_prompt_for_image_type(image_type)
        else:
            # Use the original generic prompt
            prompt = "Analyze the provided image and provide a detailed description."
            image_type = "general"
        
        # Resize image while maintaining aspect ratio
        max_dimension = 1024
        width, height = image.size
        
        # Only resize if the image is larger than the max dimension
        if width > max_dimension or height > max_dimension:
            # Calculate the scaling factor
            scale_factor = max_dimension / max(width, height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            
            # Resize the image
            image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
            if debug:
                print(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
        else:
            image_resized = image
            
        # Convert PIL Image to base64
        buffered = BytesIO()
        image_resized.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Determine max tokens based on caption length
        if caption_length == "short":
            max_tokens = 100
        elif caption_length == "detailed":
            max_tokens = 500
        else:  # normal
            max_tokens = 300
        
        # Prepare the request for Ollama
        url = f"http://localhost:{ollama_port}/api/generate"
        
        # Create the request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more precise outputs
                "max_tokens": max_tokens
            }
        }
        
        # Send the request to Ollama
        start_time = time.time()
        
        if debug:
            print(f"Sending captioning request to Ollama with {image_type} prompt...")
        
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            caption = result.get("response", "")
            
            if debug:
                print(f"Caption generated in {time.time() - start_time:.2f} seconds")
                print(f"Generated caption: {caption}")
            
            return caption
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            if debug:
                print(f"Ollama API error: {response.status_code} - {response.text}")
            return fallback_caption(image)
            
    except Exception as e:
        logger.error(f"Error in caption generation: {e}")
        if debug:
            print(f"Caption error: {e}")
        return fallback_caption(image)

def generate_caption_for_image(image, caption_length="normal", debug=False, image_type=None, use_enhanced_captions=True):
    """
    Generate a caption for an image using an advanced vision-language model
    
    Args:
        image: PIL Image object
        caption_length: Length of caption ("short", "normal", "detailed")
        debug: Whether to print debug information
        image_type: Override for image type detection
        use_enhanced_captions: Whether to use the enhanced captioning system
        
    Returns:
        Generated caption string
    """
import fitz
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import tiktoken
from abc import ABC, abstractmethod
import os
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import hashlib
import re
import time
import pdfkit
import tempfile
import aiohttp
import concurrent.futures
from functools import partial
import asyncio

logger = logging.getLogger(__name__)

class ContentProcessor(ABC):
    """
    An abstract base class for content processors.
    """
    @abstractmethod
    def extract_content(self, source: str, **kwargs) -> List[Dict]:
        """
        Extracts content from a source.

        :param source: The content source (e.g., file path, URL, text).
        :param kwargs: Additional keyword arguments for the processor.
                      Required kwargs:
                      - output_image_dir: str - Directory to save extracted images
                      - content_id: str - Unique identifier for this content
        :return: A list of dictionaries, where each dict represents a content block (e.g., text or image).
                Each dict must have a 'type' key with value either 'text' or 'image'.
                For 'text' type: {'type': 'text', 'content': str}
                For 'image' type: {'type': 'image', 'path': str, 'placeholder': str}
        """
        pass

    @abstractmethod
    def create_chunks(self, content_list: List[Dict], **kwargs) -> List[Dict]:
        """
        Creates chunks from the extracted content.

        :param content_list: The list of content blocks from extract_content.
        :param kwargs: Additional keyword arguments for chunking.
                      Optional kwargs:
                      - chunk_size: int - Target size for each chunk (default: 1024)
                      - chunk_overlap: int - Number of overlapping tokens (default: 100)
        :return: A list of chunk dictionaries, each containing:
                - text: str - The text content of the chunk
                - images: List[str] - List of image placeholders in this chunk
        """
        pass

    def _generate_image_filename(self, content_id: str, counter: int, ext: str, prefix: Optional[str] = None) -> str:
        """
        Generate a consistent image filename across all processors.

        :param content_id: Unique identifier for the content
        :param counter: Image counter/index
        :param ext: Image file extension (without dot)
        :param prefix: Optional prefix to identify source type (e.g., 'web', 'pdf')
        :return: Generated filename
        """
        if prefix:
            return f"{content_id}_{prefix}_image_{counter}.{ext}"
        return f"{content_id}_image_{counter}.{ext}"

class TextProcessor(ContentProcessor):
    """
    Processor for plain text content.
    """
    def extract_content(self, source: str, **kwargs) -> List[Dict]:
        """
        Extracts content from a text file.

        :param source: Path to the text file
        :param kwargs: Additional keyword arguments
                      Required:
                      - content_id: str - Unique identifier for this content
        :return: List of dictionaries containing text content
        """
        if 'content_id' not in kwargs:
            raise ValueError("content_id is required for text processing")

        try:
            with open(source, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split into paragraphs and create content blocks
            content_list = []
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                content_list.append({
                    "type": "text",
                    "content": para
                })
            
            return content_list
            
        except Exception as e:
            logging.error(f"Error processing text file: {e}")
            raise

    def create_chunks(self, content_list: List[Dict], **kwargs) -> List[Dict]:
        """
        Creates chunks from the extracted text content.

        :param content_list: The list of content blocks.
        :param kwargs: Expects 'chunk_size' and 'chunk_overlap'.
        :return: List of chunk dictionaries.
        """
        chunk_size = kwargs.get("chunk_size", 1024)
        chunk_overlap = kwargs.get("chunk_overlap", 100)

        logger.info("Creating text chunks...")
        # Build a flat list of content blocks
        flat_blocks = []
        for item in content_list:
            if item['type'] == 'text':
                flat_blocks.append({'type': 'text', 'content': item['content']})

        tokenizer = tiktoken.get_encoding("cl100k_base")
        chunks = []
        i = 0
        while i < len(flat_blocks):
            chunk_text = ''
            token_count = 0
            j = i
            while j < len(flat_blocks) and token_count < chunk_size:
                block = flat_blocks[j]
                block_tokens = tokenizer.encode(block['content'])
                if token_count + len(block_tokens) > chunk_size and token_count > 0:
                    break  # Don't overflow chunk, unless it's the first block
                chunk_text += block['content'] + '\n\n'
                token_count += len(block_tokens)
                j += 1
            
            # Overlap logic: back up by chunk_overlap tokens
            next_i = j
            if next_i < len(flat_blocks) and chunk_overlap > 0:
                # Find how many tokens to back up
                overlap_tokens = 0
                k = j - 1
                while k >= i and overlap_tokens < chunk_overlap:
                    block = flat_blocks[k]
                    overlap_tokens += len(tokenizer.encode(block['content']))
                    k -= 1
                next_i = max(i + 1, k + 1)
            
            chunks.append({
                'text': chunk_text.strip(),
                'images': []  # Text content has no images
            })
            i = next_i

        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

class YoutubeProcessor(ContentProcessor):
    """
    A class to process YouTube content.
    """
    def extract_content(self, source: str, **kwargs) -> List[Dict]:
        # Implementation to fetch transcript from youtube_url
        pass

    def create_chunks(self, content_list: List[Dict], **kwargs) -> List[str]:
        # Implementation to chunk the transcript
        pass

# class WebProcessor(ContentProcessor):
#     """
#     A class to process web content.
#     """
#     def __init__(self):
#         super().__init__()
#         self.supported_image_types = {
#             'image/jpeg': '.jpg',
#             'image/png': '.png',
#             'image/gif': '.gif',
#             'image/webp': '.webp'
#         }
#         self.max_retries = 3
#         self.timeout = 30  # Increased timeout for large images

#     def _is_valid_url(self, url: str) -> bool:
#         """Check if the URL is valid and has a supported scheme."""
#         try:
#             result = urlparse(url)
#             return all([result.scheme in ['http', 'https'], result.netloc])
#         except:
#             return False

#     def _download_with_retry(self, url: str) -> Optional[requests.Response]:
#         """Download with retry logic"""
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#             'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
#             'Accept-Language': 'en-US,en;q=0.9',
#             'Referer': url
#         }
        
#         for attempt in range(self.max_retries):
#             try:
#                 response = requests.get(url, headers=headers, timeout=self.timeout)
#                 response.raise_for_status()
#                 return response
#             except Exception as e:
#                 if attempt == self.max_retries - 1:
#                     raise
#                 wait_time = (attempt + 1) * 2  # Exponential backoff
#                 logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}. Retrying in {wait_time}s...")
#                 time.sleep(wait_time)
#         return None

#     def extract_content(self, source: str, **kwargs) -> List[Dict]:
#         """Extract content from a URL."""
#         if not self._is_valid_url(source):
#             raise ValueError("Invalid URL provided")

#         output_image_dir = kwargs.get("output_image_dir")
#         content_id = kwargs.get("content_id")
        
#         if not output_image_dir:
#             raise ValueError("output_image_dir is required for WebProcessor")
#         if not content_id:
#             raise ValueError("content_id is required for WebProcessor")

#         logger.info(f"Extracting content from {source}...")

#         try:
#             # Fetch the webpage
#             response = self._download_with_retry(source)
            
#             # Parse HTML
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Remove unwanted elements
#             for tag in ['script', 'style', 'noscript', 'iframe', 'svg']:
#                 for element in soup.find_all(tag):
#                     element.decompose()

#             # Extract all image URLs
#             image_urls = set()
            
#             # Get src and srcset from img tags
#             for img in soup.find_all('img'):
#                 if src := img.get('src'):
#                     image_urls.add(src)
#                 if srcset := img.get('srcset'):
#                     urls = [u.strip().split()[0] for u in srcset.split(',')]
#                     image_urls.update(urls)
            
#             # Get background images from style attributes
#             for elem in soup.find_all(lambda tag: tag.get('style')):
#                 if style := elem.get('style'):
#                     if url := re.search(r'url\([\'"]?(.*?)[\'"]?\)', style):
#                         image_urls.add(url.group(1))

#             # Make URLs absolute and filter out SVGs and tiny images
#             image_urls = {urljoin(source, img_url) for img_url in image_urls 
#                          if not img_url.lower().endswith('.svg')}
            
#             # Create content blocks
#             content_blocks = []
            
#             # Process text content block by block, similar to PDF
#             for block in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
#                 # Skip if block contains no text or only whitespace
#                 text = block.get_text(strip=True)
#                 if not text:
#                     continue
                    
#                 # Skip if block is just a container for other blocks we'll process
#                 if block.name == 'div' and len(block.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])) > 0:
#                     continue
                    
#                 content_blocks.append({
#                     'type': 'text',
#                     'content': text
#                 })
            
#             # Process images
#             successful_images = 0
#             for idx, img_url in enumerate(image_urls, 1):
#                 try:
#                     # Download image with retry
#                     img_response = self._download_with_retry(img_url)
#                     if not img_response:
#                         continue
                    
#                     # Check content type
#                     content_type = img_response.headers.get('content-type', '').lower()
#                     if content_type not in self.supported_image_types:
#                         logger.warning(f"Unsupported image type {content_type} for {img_url}")
#                         continue
                    
#                     # Generate filename and save image
#                     extension = self.supported_image_types[content_type]
#                     img_filename = self._generate_image_filename(content_id, idx, extension.lstrip('.'), 'web')
#                     img_path = os.path.join(output_image_dir, img_filename)
                    
#                     with open(img_path, 'wb') as f:
#                         f.write(img_response.content)
                    
#                     # Add image block
#                     content_blocks.append({
#                         'type': 'image',
#                         'path': img_path,
#                         'placeholder': f"\n\n[IMAGE: {img_filename}]\n\n"
#                     })
#                     successful_images += 1
                    
#                 except Exception as e:
#                     logger.warning(f"Failed to process image {img_url}: {str(e)}")
#                     continue

#             logger.info(f"Extracted {len(content_blocks)} content blocks ({successful_images} of {len(image_urls)} images downloaded successfully)")
#             return content_blocks
            
#         except requests.RequestException as e:
#             raise RuntimeError(f"Failed to fetch URL: {str(e)}")
#         except Exception as e:
#             raise RuntimeError(f"Error processing URL content: {str(e)}")

#     def create_chunks(self, content_list: List[Dict], **kwargs) -> List[Dict]:
#         """Creates chunks from the extracted web content."""
#         chunk_size = kwargs.get("chunk_size", 1024)
#         chunk_overlap = kwargs.get("chunk_overlap", 100)

#         logger.info("Creating text chunks...")
        
#         # Build a flat list of content blocks
#         flat_blocks = []
#         for item in content_list:
#             if item['type'] == 'text':
#                 flat_blocks.append({'type': 'text', 'content': item['content']})
#             elif item['type'] == 'image':
#                 flat_blocks.append({'type': 'image', 'placeholder': item['placeholder']})

#         tokenizer = tiktoken.get_encoding("cl100k_base")
#         chunks = []
#         i = 0
#         while i < len(flat_blocks):
#             chunk_text = '' 
#             chunk_images = []
#             token_count = 0
#             j = i
#             while j < len(flat_blocks) and token_count < chunk_size:
#                 block = flat_blocks[j]
#                 if block['type'] == 'text':
#                     block_tokens = tokenizer.encode(block['content'])
#                     if token_count + len(block_tokens) > chunk_size and token_count > 0:
#                         break  # Don't overflow chunk, unless it's the first block
#                     chunk_text += block['content'] + '\n\n'
#                     token_count += len(block_tokens)
#                 elif block['type'] == 'image':
#                     chunk_images.append(block['placeholder'])
#                 j += 1
            
#             # Overlap logic: back up by chunk_overlap tokens
#             next_i = j
#             if next_i < len(flat_blocks) and chunk_overlap > 0:
#                 # Find how many tokens to back up
#                 overlap_tokens = 0
#                 k = j - 1
#                 while k >= i and overlap_tokens < chunk_overlap:
#                     block = flat_blocks[k]
#                     if block['type'] == 'text':
#                         overlap_tokens += len(tokenizer.encode(block['content']))
#                     k -= 1
#                 next_i = max(i + 1, k + 1)
            
#             chunks.append({
#                 'text': chunk_text.strip(),
#                 'images': chunk_images
#             })
#             i = next_i

#         logger.info(f"Created {len(chunks)} chunks.")
#         return chunks

class PdfProcessor(ContentProcessor):
    """
    A class to extract content from a PDF file.
    """

    def extract_content(self, source: str, **kwargs) -> List[Dict]:
        """
        Extracts content from a PDF file.

        :param source: Path to the PDF file.
        :param kwargs: Expects 'output_image_dir' and 'content_id'.
        :return: List of content dictionaries.
        """
        pdf_path = source
        output_image_dir = kwargs.get("output_image_dir")
        content_id = kwargs.get("content_id")
        
        if not output_image_dir:
            raise ValueError("output_image_dir is required for PdfProcessor")
        if not content_id:
            raise ValueError("content_id is required for PdfProcessor")

        logger.info(f"Extracting content from {pdf_path}...")
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        content_list = []
        Path(output_image_dir).mkdir(parents=True, exist_ok=True)
        img_counter = 0

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text
                    text = "\n".join(
                        "".join(span["text"] for span in line["spans"]) for line in block["lines"]
                    )
                    if text.strip():
                        content_list.append({"type": "text", "content": text.strip()})
                elif block["type"] == 1:  # Image
                    try:
                        img_counter += 1
                        # Prefer xref if available
                        xref = block.get("image", {}).get("xref") if isinstance(block.get("image"), dict) else None
                        if not xref:
                            xref = block.get("xref")
                        if xref:
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                        elif isinstance(block.get("image"), bytes):
                            image_bytes = block["image"]
                            image_ext = "png"  # Default/fallback
                        else:
                            logger.warning(f"No xref or image bytes found for image on page {page_num + 1}")
                            continue
                        
                        # Use the common image filename generator
                        img_filename = self._generate_image_filename(content_id, img_counter, image_ext, "pdf")
                        img_path = Path(output_image_dir) / img_filename
                        img_path.write_bytes(image_bytes)
                        content_list.append({
                            "type": "image",
                            "path": str(img_path),
                            "placeholder": f"\n\n[IMAGE: {img_filename}]\n\n"
                        })
                    except Exception as e:
                        logger.warning(f"Could not extract image on page {page_num + 1}: {e}")
                else:
                    logger.debug(f"Skipping block type {block['type']} on page {page_num + 1}")
        doc.close()
        logger.info(f"Extracted {len(content_list)} content blocks ({img_counter} images).")
        return content_list

    def create_chunks(self, content_list: List[Dict], **kwargs) -> List[Dict]:
        """
        Creates chunks from the extracted PDF content.

        :param content_list: The list of content blocks.
        :param kwargs: Expects 'chunk_size' and 'chunk_overlap'.
        :return: List of chunk dictionaries.
        """
        chunk_size = kwargs.get("chunk_size", 1024)
        chunk_overlap = kwargs.get("chunk_overlap", 100)

        logger.info("Creating text chunks...")
        # Build a flat list of (type, content/placeholder) for easier processing
        flat_blocks = []
        for item in content_list:
            if item['type'] == 'text':
                flat_blocks.append({'type': 'text', 'content': item['content']})
            elif item['type'] == 'image':
                flat_blocks.append({'type': 'image', 'placeholder': item['placeholder']})

        tokenizer = tiktoken.get_encoding("cl100k_base")
        chunks = []
        i = 0
        while i < len(flat_blocks):
            chunk_text = ''
            chunk_images = []
            token_count = 0
            j = i
            while j < len(flat_blocks) and token_count < chunk_size:
                block = flat_blocks[j]
                if block['type'] == 'text':
                    block_tokens = tokenizer.encode(block['content'])
                    if token_count + len(block_tokens) > chunk_size and token_count > 0:
                        break  # Don't overflow chunk, unless it's the first block
                    chunk_text += block['content']
                    token_count += len(block_tokens)
                elif block['type'] == 'image':
                    chunk_images.append(block['placeholder'])
                j += 1
            # Overlap logic: back up by chunk_overlap tokens, but not into the middle of an image
            next_i = j
            if next_i < len(flat_blocks) and chunk_overlap > 0:
                # Find how many tokens to back up
                overlap_tokens = 0
                k = j - 1
                while k >= i and overlap_tokens < chunk_overlap:
                    block = flat_blocks[k]
                    if block['type'] == 'text':
                        overlap_tokens += len(tokenizer.encode(block['content']))
                    k -= 1
                next_i = max(i + 1, k + 1)
            chunks.append({'text': chunk_text, 'images': chunk_images})
            i = next_i
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

# class WebToPdfProcessor(ContentProcessor):
#     """
#     A class to process web content by first converting it to PDF using Playwright.
#     This provides consistent processing with PDF content.
#     """
#     def __init__(self):
#         super().__init__()
#         self.pdf_processor = PdfProcessor()
#         # Create a thread pool executor
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

#     def _is_valid_url(self, url: str) -> bool:
#         """Check if the URL is valid and has a supported scheme."""
#         try:
#             result = urlparse(url)
#             return all([result.scheme in ['http', 'https'], result.netloc])
#         except:
#             return False
        
#     def _is_valid_pdf(self, url: str) -> bool:
#         """
#         Check if the URL points directly to a PDF file
        
#         Args:
#             url: The URL to check
#         Returns:
#             bool: True if the URL points to a PDF, False otherwise
#         """
#         try:
#             # Send a HEAD request first to check content type
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#             }
#             response = requests.head(url, headers=headers, allow_redirects=True)
            
#             # Check content type header
#             content_type = response.headers.get('Content-Type', '').lower()
#             if 'application/pdf' in content_type:
#                 return True
                
#             # Some servers might not set content type correctly
#             # Check if URL ends with .pdf
#             if url.lower().endswith('.pdf'):
#                 # Verify by downloading first few bytes
#                 response = requests.get(url, headers=headers, stream=True)
#                 first_bytes = next(response.iter_content(256))
#                 # Check for PDF magic number (%PDF-)
#                 if first_bytes.startswith(b'%PDF-'):
#                     return True
                    
#             return False
            
#         except Exception as e:
#             logger.warning(f"Error checking if URL is PDF: {e}")
#             return False

#     def _download_pdf(self, url: str, output_path: str) -> None:
#         """
#         Download a PDF file from URL
        
#         Args:
#             url: The URL of the PDF
#             output_path: Where to save the PDF
#         """
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         }
        
#         try:
#             with requests.get(url, headers=headers, stream=True) as response:
#                 response.raise_for_status()
#                 with open(output_path, 'wb') as f:
#                     for chunk in response.iter_content(chunk_size=8192):
#                         f.write(chunk)
                        
#             logger.info(f"PDF downloaded to: {output_path}")
            
#         except Exception as e:
#             raise RuntimeError(f"Failed to download PDF: {e}")

#     def extract_content(self, source: str, **kwargs) -> List[Dict]:
#         """
#         Extract content from a URL by converting to PDF first.
        
#         :param source: The URL to process
#         :param kwargs: Additional arguments (output_image_dir and content_id required)
#         :return: List of content blocks (text and images)
#         """
#         if not self._is_valid_url(source):
#             raise ValueError("Invalid URL provided")

#         output_image_dir = kwargs.get("output_image_dir")
#         content_id = kwargs.get("content_id")
        
#         if not output_image_dir:
#             raise ValueError("output_image_dir is required for WebToPdfProcessor")
#         if not content_id:
#             raise ValueError("content_id is required for WebToPdfProcessor")

#         logger.info(f"Processing web content from {source}...")

#         # Save PDF in uploads directory
#         pdf_path = os.path.join("uploads", f"{content_id}.pdf")
        
#         try:
#             # Check if URL is direct PDF
#             if self._is_valid_pdf(source):
#                 logger.info("URL points to a PDF file, downloading directly...")
#                 self._download_pdf(source, pdf_path)
#                 title = os.path.basename(source)  # Use filename as title for PDFs
#             else:
#                 # Convert webpage to PDF using Playwright
#                 logger.info(f"Converting web page to PDF...")
#                 title = webpage_to_pdf(source, pdf_path)
            
#             if not os.path.exists(pdf_path):
#                 raise RuntimeError("PDF conversion/download failed")
            
#             # Use PdfProcessor to handle the converted PDF
#             logger.info("Processing converted PDF...")
#             content_list = self.pdf_processor.extract_content(
#                 pdf_path,
#                 output_image_dir=output_image_dir,
#                 content_id=content_id
#             )

#             # Add the title as metadata if we got one
#             if content_list and title:
#                 if content_list[0]['type'] == 'text':
#                     content_list[0]['metadata'] = {'title': title}
#                 else:
#                     content_list.insert(0, {
#                         'type': 'text',
#                         'content': '',
#                         'metadata': {'title': title}
#                     })

#             return content_list

#         except Exception as e:
#             # Clean up PDF if conversion fails
#             if os.path.exists(pdf_path):
#                 try:
#                     os.remove(pdf_path)
#                 except:
#                     pass
#             logger.error(f"Error processing web content: {str(e)}")
#             raise RuntimeError(f"Failed to process web content: {str(e)}")

#     def create_chunks(self, content_list: List[Dict], **kwargs) -> List[Dict]:
#         """Use PdfProcessor's chunking logic"""
#         return self.pdf_processor.create_chunks(content_list, **kwargs)

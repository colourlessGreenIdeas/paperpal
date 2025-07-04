import fitz
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import tiktoken
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from functools import partial

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

            # Return the entire text as a single block
            return [{
                "type": "text",
                "content": text.strip()
            }]
            
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
        
        if not content_list or not content_list[0]["content"]:
            return []

        # Get the full text
        full_text = content_list[0]["content"]
        
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Get all tokens
        tokens = tokenizer.encode(full_text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk_size tokens
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode chunk
            chunk_text = tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text.strip(),
                'images': []  # Text content has no images
            })
            
            # Move to next chunk, accounting for overlap
            start_idx = end_idx - chunk_overlap
            
            # If we're near the end and the remaining text is small, include it in the last chunk
            if len(tokens) - start_idx < chunk_size / 2:
                break

        logger.info(f"Created {len(chunks)} chunks from {len(tokens)} tokens.")
        return chunks

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

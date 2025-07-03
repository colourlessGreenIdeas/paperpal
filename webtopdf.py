from playwright.sync_api import sync_playwright
import time
import concurrent.futures
import threading
from typing import Optional
import logging
import requests
import os

# Configure logging
logger = logging.getLogger(__name__)

# Create a thread-local storage for the playwright browser
thread_local = threading.local()

def _get_browser():
    """Get or create a browser instance for the current thread"""
    if not hasattr(thread_local, "browser"):
        p = sync_playwright().start()
        thread_local.playwright = p
        thread_local.browser = p.chromium.launch(
            # Browser configuration
            args=[
                '--disable-web-security',  # Disable CORS for better compatibility
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',  # Prevent memory issues
            ]
        )
    return thread_local.browser

def _cleanup_browser():
    """Clean up browser and playwright for the current thread"""
    if hasattr(thread_local, "browser"):
        try:
            thread_local.browser.close()
            thread_local.playwright.stop()
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")
        finally:
            del thread_local.browser
            del thread_local.playwright

# Create a single thread pool for all PDF conversions
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _is_valid_pdf(url: str) -> bool:
    """
    Check if the URL points directly to a PDF file
    
    Args:
        url: The URL to check
    Returns:
        bool: True if the URL points to a PDF, False otherwise
    """
    try:
        # Send a HEAD request first to check content type
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.head(url, headers=headers, allow_redirects=True)
        
        # Check content type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type:
            return True
            
        # Some servers might not set content type correctly
        # Check if URL ends with .pdf
        if url.lower().endswith('.pdf'):
            # Verify by downloading first few bytes
            response = requests.get(url, headers=headers, stream=True)
            first_bytes = next(response.iter_content(256))
            # Check for PDF magic number (%PDF-)
            if first_bytes.startswith(b'%PDF-'):
                return True
                
        return False
        
    except Exception as e:
        logger.warning(f"Error checking if URL is PDF: {e}")
        return False

def _download_pdf(url: str, output_path: str) -> Optional[str]:
    """
    Download a PDF file from URL
    
    Args:
        url: The URL of the PDF
        output_path: Where to save the PDF
    Returns:
        str: The filename as the title
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        logger.info(f"PDF downloaded to: {output_path}")
        return os.path.basename(url)  # Use filename as title for PDFs
        
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise

def _convert_in_thread(url: str, output_path: str, wait_time: int = 5, max_retries: int = 3) -> Optional[str]:
    """
    The actual conversion function that runs in a separate thread
    Returns the page title if successful
    """
    try:
        browser = _get_browser()
        context = browser.new_context(
            viewport={'width': 1280, 'height': 1024},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        )
        page = context.new_page()
        
        # Set extra headers for better compatibility
        page.set_extra_http_headers({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1'
        })

        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Navigate to the page with a longer timeout
                page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait for JavaScript to execute
                page.wait_for_timeout(wait_time * 1000)
                
                # Wait for common content selectors
                selectors = ['article', 'main', '.content', '#content']
                for selector in selectors:
                    try:
                        page.wait_for_selector(selector, timeout=5000)
                        break
                    except:
                        continue
                
                # Get the page title
                title = page.title()
                
                # Generate PDF
                page.pdf(
                    path=output_path,
                    format='A4',
                    print_background=True,
                    margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'},
                    scale=0.9  # Slightly reduce scale to fit more content
                )
                
                logger.info(f"PDF saved to: {output_path}")
                return title
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Attempt {retry_count} failed: {e}. Retrying...")
                    time.sleep(retry_count * 2)  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                    raise
                
        if last_error:
            raise last_error
            
    except Exception as e:
        logger.error(f"Error during PDF conversion: {e}")
        raise
    finally:
        try:
            page.close()
            context.close()
        except:
            pass
        _cleanup_browser()

def webpage_to_pdf(url: str, output_path: str, wait_time: int = 5) -> Optional[str]:
    """
    Convert webpage to PDF using Playwright in a separate thread
    
    Args:
        url: URL of the webpage
        output_path: Path where PDF will be saved
        wait_time: Time to wait for JavaScript to load (seconds)
    Returns:
        str: The page title if successful, None otherwise
    """
    try:
        # First check if URL points to a PDF
        if _is_valid_pdf(url):
            logger.info("URL points to a PDF file, downloading directly...")
            return _download_pdf(url, output_path)
            
        # Otherwise convert webpage to PDF
        future = _thread_pool.submit(_convert_in_thread, url, output_path, wait_time)
        return future.result()  # Wait for the conversion to complete
    except Exception as e:
        logger.error(f"Failed to convert webpage to PDF: {e}")
        raise

# Clean up the thread pool on program exit
import atexit
atexit.register(lambda: _thread_pool.shutdown(wait=True))

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    webpage_to_pdf("https://example.com", "output.pdf")
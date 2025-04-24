import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI, AzureOpenAI
from google import genai
from google.genai import types
import tiktoken
import time
import logging
import json
from typing import List, Dict, Optional
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
# load_dotenv()


# Explicitly load .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')  # Looks in script's directory
load_dotenv(env_path)  # Load specific file

# Verify loading
print("API Key exists:", "OPENAI_API_KEY" in os.environ)  # Should print True
print("Gemini API Key exists:", "GOOGLE_API_KEY" in os.environ)  # Should print True if using Gemini
print("Model:", os.getenv("OPENAI_MODEL", "NOT FOUND"))  # Should show your model
print("Gemini Model:", os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001"))  # Default Gemini model

class RateLimiter:
    """Handles rate limiting across multiple threads"""
    def __init__(self, max_calls_per_minute: int = 3000):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls_made = 0
        self.last_reset_time = time.time()
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_reset_time >= 60:
                self.calls_made = 0
                self.last_reset_time = current_time
                
            if self.calls_made >= self.max_calls_per_minute:
                sleep_time = 60 - (current_time - self.last_reset_time)
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                self.calls_made = 0
                self.last_reset_time = time.time()
                
            self.calls_made += 1

class Paperpal:
    def __init__(self, model_temperature: float = 0.3, provider: str = "openai"):
        """Initialize the Paperpal with OpenAI, Azure OpenAI, or Google Gemini configuration.
        
        Args:
            model_temperature (float): Temperature for the language model
            provider (str): AI provider to use - 'openai', 'azure', or 'gemini' (default: 'openai')
        """
        try:
            # Set provider
            self.provider = provider.lower()
            logger.info(f"Using AI provider: {self.provider}")
            
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
                
                # Validate API key format
                if len(api_key) < 40:
                    raise ValueError(f"OpenAI API key seems too short (length: {len(api_key)}). Please check your .env file format.")
                if not api_key.startswith('sk-'):
                    raise ValueError("OpenAI API key must start with 'sk-'. Please check your .env file.")
                
                logger.info(f"API key validation passed. Length: {len(api_key)}")
                self.client = OpenAI(api_key=api_key)
                self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview").strip()
                self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
                logger.info(f"OpenAI Model: {self.model}")
                
                
            elif self.provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY", "").strip()
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is required for Gemini provider")
                
                # Initialize Gemini client
                # genai.configure(api_key=api_key)
                # Only run this block for Gemini Developer API
                self.client = genai.Client(api_key=api_key)
                # self.client = genai.Client()
                self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001").strip()
                logger.info(f"Gemini Model: {self.model}")
                
            else:  # azure
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
                api_version = os.getenv("AZURE_API_VERSION")
                
                if not all([api_key, api_base, api_version]):
                    raise ValueError("Missing required Azure OpenAI configuration")

                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=api_base
                )
                
                self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
                self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
                if not self.deployment_name:
                    raise ValueError("AZURE_DEPLOYMENT_NAME is required")
            
            self.temperature = model_temperature
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Define chunk parameters
            self.chunk_size = 1500
            self.chunk_overlap = 400
            
            # Define grade level descriptions
            self.grade_level_descriptions = {
                "original": "return the original text without any simplification",
                "grade1": "a 6-7 year old first grader, using very simple words and concepts",
                "grade2": "a 7-8 year old second grader, using simple words and basic concepts",
                "grade3": "an 8-9 year old third grader, introducing slightly more complex concepts",
                "grade4": "a 9-10 year old fourth grader, using more advanced vocabulary",
                "grade5": "a 10-11 year old fifth grader, introducing basic scientific concepts",
                "grade6": "an 11-12 year old sixth grader, using more scientific terminology",
                "grade7": "a 12-13 year old seventh grader, introducing pre-algebra concepts",
                "grade8": "a 13-14 year old eighth grader, using algebra-level math concepts",
                "grade9": "a 14-15 year old ninth grader, introducing basic high school concepts",
                "grade10": "a 15-16 year old tenth grader, using more advanced high school concepts",
                "grade11": "a 16-17 year old eleventh grader, preparing for college-level work",
                "grade12": "a 17-18 year old twelfth grader, at advanced placement level",
                "freshman": "a college freshman, introducing undergraduate-level concepts",
                "undergraduate": "a general undergraduate student with basic field knowledge",
                "bachelors": "a graduating bachelor's student with solid field knowledge",
                "masters": "a master's level student with advanced field knowledge",
                "phd": "a PhD level researcher with expert field knowledge"
            }
            
            # Initialize rate limiter
            self.rate_limiter = RateLimiter(max_calls_per_minute=3000)
            
        except Exception as e:
            logger.error(f"Failed to initialize paperpal: {str(e)}")
            raise
    
    def get_grade_level_description(self, grade_level: str) -> str:
        """Get the description for the specified grade level."""
        return self.grade_level_descriptions.get(
            grade_level.lower().replace("-", "").replace(" ", ""),
            "a general undergraduate student with basic field knowledge"
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - self.chunk_overlap
            
        return chunks
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with error handling."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def get_completion(self, prompt: str) -> str:
        """Get completion from OpenAI, Azure OpenAI, or Gemini with rate limiting."""
        self.rate_limiter.wait_if_needed()
        
        try:
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        system_instruction="""You are an expert academic researcher who excels at ensuring accuracy of complex content.
                    When handling mathematical equations:
                    1. Use $ for inline equations (e.g., $x = y$)
                    2. Use $$ for display equations (e.g., $$\\sum_{i=1}^n x_i$$)
                    3. Never use \\( or \\) or \\[ or \\] for equations
                    4. Use proper LaTeX notation for mathematical symbols
                    5. Keep variable names and mathematical symbols consistent"""
                    )
                )
                return response.text
            else:
                messages = [
                    {"role": "system", "content": """You are an expert academic researcher who excels at ensuring accuracy of complex content.
                    When handling mathematical equations:
                    1. Use $ for inline equations (e.g., $x = y$)
                    2. Use $$ for display equations (e.g., $$\\sum_{i=1}^n x_i$$)
                    3. Never use \\( or \\) or \\[ or \\] for equations
                    4. Use proper LaTeX notation for mathematical symbols
                    5. Keep variable names and mathematical symbols consistent"""},
                    {"role": "user", "content": prompt}
                ]

                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature
                    )
                else:  # azure
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        temperature=self.temperature
                    )
                
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            raise
    
    def simplify_chunk(self, chunk: str, context: str, grade_level: str) -> str:
        """Simplify a specific chunk of text for a specific grade level while maintaining academic integrity."""
        try:
            grade_desc = self.get_grade_level_description(grade_level)
            
            if grade_level == "original":
                prompt = f"""For the given context, return the original text without any simplification. Follow the original text exactly. Here are additional instructions:
                - Do not change the text
                - Do not add any additional text
                - Do not remove any text
                - Do not change the order of the text
                - Do not change the structure of the text
                - Do not change the formatting of the text
                - Do not change the mathematical content of the text
                - Do not change the scientific content of the text
                - Do not change the technical content of the text
                - Do not change the academic content of the text
                - Do not change the professional content of the text
                - Do not change the technical content of the text
                Current Chunk to Simplify:
                {chunk}
                
                Processed version:
                """
            else:
                prompt = f"""Using the provided context and specific chunk, create a simplified version that would be suitable for {grade_desc}. The simplified version should:
                1. Use language and explanations appropriate for {grade_desc}
                2. Explain complex concepts in terms that {grade_desc} would understand
                3. Preserve important information while adjusting the complexity level
                4. Maintain consistency with the overall paper context
                5. If equations are present:
                - For grade 1-8: Focus on conceptual understanding with minimal equations
                - For grade 9-12: Include simplified equations with clear explanations
                - For undergraduate and above: Include full equations with appropriate explanations
                6. For tables, simplify them to a level appropriate for {grade_desc}
                7. Include examples or analogies suitable for {grade_desc} when helpful
                8. Do not repeat explanations that were already covered in previous sections
                9. If a concept was already explained in a previous section, you can refer to it briefly
                10. Focus on new information and concepts not covered in previous sections

                Overall Paper Context and Previously Simplified Content:
                {context}
                
                Current Chunk to Simplify:
                {chunk}
                
                Simplified version:"""
            
            return self.get_completion(prompt)
        except Exception as e:
            logger.error(f"Error simplifying chunk for grade {grade_level}: {str(e)}")
            raise
    
    def process_chunk_for_grade(self, chunk: str, context: str, grade_level: str, result_dict: dict, chunk_index: int):
        """Process a single chunk for a specific grade level and store the result."""
        try:
            simplified = self.simplify_chunk(chunk, context, grade_level)
            result_dict[grade_level][chunk_index] = simplified
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} for grade {grade_level}: {str(e)}")
            result_dict[grade_level][chunk_index] = f"ERROR: {str(e)}"
    
    def process_all_grade_levels(self, chunks: List[str], grade_levels: List[str]) -> Dict[str, List[str]]:
        """Process all chunks for all grade levels in parallel."""
        results = {grade: [""] * len(chunks) for grade in grade_levels}
        total_chunks = len(chunks)
        
        with ThreadPoolExecutor(max_workers=min(5, len(grade_levels))) as executor:
            futures = []
            
            for i, chunk in enumerate(chunks):
                # Create context from original text start and recent chunks
                context = chunks[0][:1000]  # First 1000 chars of first chunk for high-level context
                if i > 0:
                    context += "\n\nRecently processed content:\n" + "\n\n".join(chunks[max(0, i-2):i])
                
                # Submit tasks for all grade levels for this chunk
                for grade_level in grade_levels:
                    futures.append(
                        executor.submit(
                            self.process_chunk_for_grade,
                            chunk=chunk,
                            context=context,
                            grade_level=grade_level,
                            result_dict=results,
                            chunk_index=i
                        )
                    )
                
                # Progress reporting
                if (i + 1) % 5 == 0 or (i + 1) == total_chunks:
                    logger.info(f"Submitted {i+1}/{total_chunks} chunks for processing")
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise exceptions if any occurred
                except Exception as e:
                    logger.error(f"Error in processing future: {str(e)}")
        
        return results
    
    def save_grade_level_output(self, simplified_content: Dict[str, List[str]], output_dir: str, base_name: str):
        """Save output for all grade levels in both JSON and Markdown formats."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual grade level files
        for grade_level, chunks in simplified_content.items():
            # Save Markdown
            md_path = os.path.join(output_dir, f"{base_name}_{grade_level}.md")
            self._save_markdown(chunks, md_path, grade_level)
            
            # Save JSON
            json_path = os.path.join(output_dir, f"{base_name}_{grade_level}.json")
            self._save_json(chunks, json_path, grade_level)
        
        # Generate HTML comparison file
        html_path = os.path.join(output_dir, f"{base_name}_comparison.html")
        self._generate_html_comparison(simplified_content, html_path, base_name)
        
    def _save_markdown(self, chunks: List[str], output_path: str, grade_level: str):
        """Save content to a formatted Markdown file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write title
                f.write(f"# Simplified Academic Paper - {grade_level}\n\n")
                
                # Add math rendering note at the top
                f.write("> **Note**: This document contains mathematical equations in LaTeX format. ")
                f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.\n\n")
                
                # Write executive summary
                f.write("## Executive Summary\n\n")
                summary = chunks[0] if chunks else ""
                summary = (summary.replace("\\(", "$")
                                .replace("\\)", "$")
                                .replace("\\[", "$$")
                                .replace("\\]", "$$"))
                f.write(f"{summary}\n\n")
                
                # Write main content
                f.write("## Main Content\n\n")
                for chunk in chunks[1:]:
                    processed_chunk = (chunk.replace("\\(", "$")
                                         .replace("\\)", "$")
                                         .replace("\\[", "$$")
                                         .replace("\\]", "$$"))
                    
                    # Handle equation blocks
                    lines = processed_chunk.split('\n')
                    formatted_lines = []
                    in_equation_block = False
                    
                    for line in lines:
                        if any(marker in line.lower() for marker in ["equation:", "formula:", "where:", "s.t.:", "subject to:"]):
                            if not in_equation_block:
                                formatted_lines.append("\n$$")
                                in_equation_block = True
                            clean_line = (line.replace("equation:", "")
                                            .replace("formula:", "")
                                            .replace("where:", "\\text{where }")
                                            .strip())
                            formatted_lines.append(clean_line)
                        else:
                            if in_equation_block and line.strip():
                                formatted_lines.append(line.strip())
                            elif in_equation_block:
                                formatted_lines.append("$$\n")
                                in_equation_block = False
                            else:
                                formatted_lines.append(line)
                    
                    if in_equation_block:
                        formatted_lines.append("$$\n")
                    
                    processed_chunk = '\n'.join(formatted_lines)
                    f.write(f"{processed_chunk}\n\n")
                
                # Add footnote about math rendering
                f.write("---\n\n")
                f.write("*Note: This document uses KaTeX/MathJax compatible math notation. ")
                f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.*\n")
                
            logger.info(f"Saved {grade_level} markdown to {output_path}")
        except Exception as e:
            logger.error(f"Error saving {grade_level} markdown: {str(e)}")
            raise
    
    def _save_json(self, chunks: List[str], output_path: str, grade_level: str):
        """Save content to a JSON file."""
        try:
            content = {
                "metadata": {
                    "grade_level": grade_level,
                    "total_chunks": len(chunks),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "chunks": chunks
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {grade_level} JSON to {output_path}")
        except Exception as e:
            logger.error(f"Error saving {grade_level} JSON: {str(e)}")
            raise
    
    def _generate_html_comparison(self, simplified_content: Dict[str, List[str]], output_path: str, base_name: str):
        """Generate an HTML file comparing all grade levels."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Comparison: """ + base_name + """</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        h1 { text-align: center; margin-bottom: 30px; }
        .grade-section { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
        .grade-title { color: #2c3e50; margin-bottom: 10px; }
        .chunk { margin-bottom: 20px; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .chunk-number { font-weight: bold; color: #7f8c8d; margin-bottom: 5px; }
        nav { position: fixed; top: 0; right: 0; background: white; padding: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        nav a { display: block; margin: 5px 0; color: #3498db; text-decoration: none; }
        nav a:hover { text-decoration: underline; }
        .content { max-width: 900px; margin: 0 auto; }
    </style>
</head>
<body>
    <nav>
        <h3>Grade Levels</h3>
""")
                
                # Write navigation links
                for grade in simplified_content.keys():
                    f.write(f'        <a href="#grade-{grade}">{grade}</a>\n')
                
                f.write("""
    </nav>
    <div class="content">
        <h1>Paper Simplification Comparison: """ + base_name + """</h1>
""")
                
                # Write content for each grade level
                for grade_level, chunks in simplified_content.items():
                    f.write(f'        <div class="grade-section" id="grade-{grade_level}">\n')
                    f.write(f'            <h2 class="grade-title">{grade_level} - {self.get_grade_level_description(grade_level)}</h2>\n')
                    
                    for i, chunk in enumerate(chunks):
                        processed_chunk = (chunk.replace("\\(", "$")
                                             .replace("\\)", "$")
                                             .replace("\\[", "$$")
                                             .replace("\\]", "$$"))
                        
                        f.write(f'            <div class="chunk">\n')
                        f.write(f'                <div class="chunk-number">Chunk {i+1}</div>\n')
                        f.write(f'                <div>{processed_chunk}</div>\n')
                        f.write('            </div>\n')
                    
                    f.write('        </div>\n')
                
                f.write("""
    </div>
</body>
</html>
""")
            
            logger.info(f"Saved HTML comparison to {output_path}")
        except Exception as e:
            logger.error(f"Error generating HTML comparison: {str(e)}")
            raise
    
    def process_paper(self, input_pdf_path: str, output_dir: str = "output", grade_levels: List[str] = None):
        """Main method to process and simplify an academic paper at all grade levels."""
        try:
            if grade_levels is None:
                grade_levels = list(self.grade_level_descriptions.keys())
            
            logger.info(f"Starting paper processing for grade levels: {', '.join(grade_levels)}")
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self.extract_text_from_pdf(input_pdf_path)
            
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            chunks = self.split_text(text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process all grade levels in parallel
            logger.info("Processing all grade levels...")
            simplified_content = self.process_all_grade_levels(chunks, grade_levels)
            
            # Save output
            base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
            self.save_grade_level_output(simplified_content, output_dir, base_name)
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Simplify academic papers while preserving mathematical content')
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('-o', '--output-dir', default='output', help='Output directory (default: output)')
    parser.add_argument('-g', '--grade-levels', 
                       nargs='+',
                       choices=['original', 'grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6', 'grade7', 'grade8',
                               'grade9', 'grade10', 'grade11', 'grade12', 'freshman', 'undergraduate', 
                               'bachelors', 'masters', 'phd'],
                       default=['original', 'grade8', 'undergraduate', 'phd'],
                       help='Target grade levels for simplification (default: original grade8 undergraduate phd)')
    parser.add_argument('-p', '--provider',
                       choices=['openai', 'azure', 'gemini'],
                       default='openai',
                       help='AI provider to use (default: openai)')
    args = parser.parse_args()
    
    try:
        # Initialize the simplifier with provider
        simplifier = Paperpal(provider=args.provider)
        
        # Process paper for all specified grade levels
        simplifier.process_paper(
            input_pdf_path=args.input,
            output_dir=args.output_dir,
            grade_levels=args.grade_levels if hasattr(args, 'grade_levels') else None
        )
        
        logger.info("Processing complete!")
        logger.info(f"Output saved to directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
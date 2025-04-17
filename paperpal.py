import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI, AzureOpenAI
from fpdf import FPDF
import tiktoken
import time
import logging
import json
from typing import List, Optional
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Paperpal:
    def __init__(self, model_temperature: float = 0.3, grade_level: str = "undergraduate", provider: str = "openai"):
        """Initialize the Paperpal with OpenAI or Azure OpenAI configuration.
        
        Args:
            model_temperature (float): Temperature for the language model
            grade_level (str): Target grade level for simplification
            provider (str): AI provider to use - 'openai' or 'azure' (default: 'openai')
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
            self.grade_level = grade_level
            
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
            
        except Exception as e:
            logger.error(f"Failed to initialize paperpal: {str(e)}")
            raise
    
    def get_grade_level_description(self) -> str:
        """Get the description for the current grade level."""
        return self.grade_level_descriptions.get(
            self.grade_level.lower().replace("-", "").replace(" ", ""),
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
        """Get completion from OpenAI or Azure OpenAI."""
        try:
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
    
    def simplify_chunk(self, chunk: str, context: str) -> str:
        """Simplify a specific chunk of text while maintaining academic integrity."""
        try:
            grade_desc = self.get_grade_level_description()
            
            if grade_desc == "original":
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
            logger.error(f"Error simplifying chunk: {str(e)}")
            raise
    
    def create_simplified_pdf(self, simplified_chunks: List[str], output_path: str):
        """Create a PDF from the simplified chunks with proper formatting.
            This does not work as expected, so not using it"""
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Add title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Simplified Academic Paper", ln=True, align='C')
            pdf.ln(10)
            
            # Add initial summary with different styling
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 10, "Executive Summary", 0, 'L')
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, simplified_chunks[0])
            pdf.ln(10)
            
            # Add main content
            pdf.set_font("Arial", "", 12)
            for chunk in simplified_chunks[1:]:
                lines = chunk.split('\n')
                for line in lines:
                    if line.strip():  # Only process non-empty lines
                        pdf.multi_cell(0, 10, line.strip())
                pdf.ln(5)
            
            pdf.output(output_path)
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            raise
    
    def save_content_to_json(self, simplified_chunks: List[str], output_path: str):
        """Save the simplified content to a JSON file.
            Currently not used, but could be useful for future reference
            """
        
        try:
            content = {
                "summary": simplified_chunks[0],
                "chunks": simplified_chunks[1:],
                "metadata": {
                    "total_chunks": len(simplified_chunks),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully saved content to {output_path}")
        except Exception as e:
            logger.error(f"Error saving content to JSON: {str(e)}")
            raise

    def save_content_to_text(self, simplified_chunks: List[str], output_path: str):
        """Save the simplified content to a formatted Markdown file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write title
                f.write("# Simplified Academic Paper\n\n")
                
                # Add math rendering note at the top
                f.write("> **Note**: This document contains mathematical equations in LaTeX format. ")
                f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.\n\n")
                
                # Write executive summary
                f.write("## Executive Summary\n\n")
                # Process summary and convert equation delimiters
                summary = simplified_chunks[0]
                # Convert equation delimiters
                summary = (summary.replace("\\(", "$")
                                .replace("\\)", "$")
                                .replace("\\[", "$$")
                                .replace("\\]", "$$"))
                f.write(f"{summary}\n\n")
                
                # Write main content
                f.write("## Main Content\n\n")
                for i, chunk in enumerate(simplified_chunks[1:], 1):
                    
                    # Convert equation delimiters in the chunk
                    processed_chunk = (chunk.replace("\\(", "$")
                                         .replace("\\)", "$")
                                         .replace("\\[", "$$")
                                         .replace("\\]", "$$"))
                    
                    # Handle any equations that might be marked with "equation:" or similar
                    lines = processed_chunk.split('\n')
                    formatted_lines = []
                    in_equation_block = False
                    
                    for line in lines:
                        # Check for equation markers
                        if any(marker in line.lower() for marker in ["equation:", "formula:", "where:", "s.t.:", "subject to:"]):
                            if not in_equation_block:
                                formatted_lines.append("\n$$")
                                in_equation_block = True
                            # Clean up the line
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
                
            logger.info(f"Successfully saved content to {output_path}")
        except Exception as e:
            logger.error(f"Error saving content to markdown file: {str(e)}")
            raise

    def process_paper(self, input_pdf_path: str, output_text_path: str = None):
        """Main method to process and simplify an academic paper."""
        try:
            logger.info("Starting paper processing...")
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self.extract_text_from_pdf(input_pdf_path)
            
            # Initialize output paths
            if output_text_path:
                # Create JSON path from markdown path
                output_json_path = str(Path(output_text_path).with_suffix('.json'))
                
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    # Write initial markdown content
                    f.write("# Simplified Academic Paper\n\n")
                    f.write("> **Note**: This document contains mathematical equations in LaTeX format. ")
                    f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.\n\n")
                    f.write("## Content\n\n")
            
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            chunks = self.split_text(text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process each chunk
            logger.info("Processing chunks...")
            simplified_chunks = []  # Keep track of previously simplified chunks
            original_chunks = []    # Keep track of original chunks
            processed_chunks = []  # Keep track of processed chunks
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)}...")
                
                # Store original chunk
                original_chunks.append(chunk)
                
                # Create context from original text start and recent simplified chunks
                context = text[:1000]  # First 1000 chars of original for high-level context
                if simplified_chunks:
                    # Add the last 2 simplified chunks as recent context
                    recent_context = "\n\n".join(simplified_chunks[-2:])
                    context = f"{context}\n\nRecently simplified content:\n{recent_context}"
                
                simplified_chunk = self.simplify_chunk(chunk, context)
                simplified_chunks.append(simplified_chunk)
                
                # Update markdown file in real-time if requested
                if output_text_path:
                    with open(output_text_path, 'a', encoding='utf-8') as f:
                        # Process chunk for markdown
                        processed_chunk = simplified_chunk
                        processed_chunk = (processed_chunk.replace("\\(", "$")
                                                        .replace("\\)", "$")
                                                        .replace("\\[", "$$")
                                                        .replace("\\]", "$$"))
                        
                        # Handle equations
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
                        processed_chunks.append(processed_chunk)
                        f.write(f"{processed_chunk}\n\n")
                
                # Respect rate limits
                time.sleep(float(os.getenv("MIN_TIME_BETWEEN_CALLS", "0.2")))
            
            # Save JSON output if path is provided
            if output_text_path:
                # Create JSON structure
                json_content = {
                    "metadata": {
                        "total_chunks": len(chunks),
                        "grade_level": self.grade_level,
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source_file": input_pdf_path
                    },
                    "chunks": [
                        {
                            "index": i,
                            "original": orig.strip(),
                            "simplified": proc.strip()
                        }
                        for i, (orig, proc) in enumerate(zip(original_chunks, processed_chunks))
                    ]
                }
                
                # Save JSON
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved JSON output to {output_json_path}")
            
            # Add final markdown footer if needed
            if output_text_path:
                with open(output_text_path, 'a', encoding='utf-8') as f:
                    f.write("\n---\n\n")
                    f.write("*Note: This document uses KaTeX/MathJax compatible math notation. ")
                    f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.*\n")
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Simplify academic papers while preserving mathematical content')
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('-g', '--grade-level', 
                       choices=['original', 'grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6', 'grade7', 'grade8',
                               'grade9', 'grade10', 'grade11', 'grade12', 'freshman', 'undergraduate', 
                               'bachelors', 'masters', 'phd'],
                       default='original',
                       help='Target grade level for simplification (default: original)')
    parser.add_argument('-p', '--provider',
                       choices=['openai', 'azure'],
                       default='openai',
                       help='AI provider to use (default: openai)')
    args = parser.parse_args()
    
    try:
        # Initialize the simplifier with grade level and provider
        simplifier = Paperpal(
            grade_level=args.grade_level,
            provider=args.provider
        )
        
        # Get input file path and create output path
        input_pdf = args.input
        base_name = os.path.splitext(input_pdf)[0]
        
        # Generate output filename
        output_text = f"{base_name}_simplified_{args.grade_level}.md"
        
        logger.info(f"Processing {input_pdf}")
        logger.info(f"Target grade level: {args.grade_level}")
        logger.info(f"Using AI provider: {args.provider}")
        logger.info(f"Output file will be:")
        logger.info(f"  - Markdown: {output_text}")
        
        # Process paper and save to markdown
        simplifier.process_paper(
            input_pdf, 
            output_text_path=output_text
        )
        
        logger.info("Processing complete!")
        logger.info(f"Simplified content saved to:")
        logger.info(f"  - {output_text}")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
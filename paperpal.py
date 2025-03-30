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
                {"role": "system", "content": """You are an expert academic researcher who excels at making complex content accessible while maintaining accuracy.
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
    
    def get_initial_summary(self, text: str) -> str:
        """Generate an initial comprehensive summary of the entire paper."""
        try:
            grade_desc = self.get_grade_level_description()
            prompt = f"""Given the following academic paper, create a comprehensive summary that would be suitable for {grade_desc}. The summary should:
            1. Capture the main points, methodology, results, and conclusions
            2. Maintain accuracy while adjusting complexity for the target level
            3. Preserve key numerical data and findings, explaining them at an appropriate level
            4. Provide context suitable for {grade_desc}
            5. If equations are present, explain them in a way that {grade_desc} would understand
            6. Use vocabulary and concepts appropriate for {grade_desc}
            
            Paper text:
            {text}
            
            Comprehensive Summary:"""
            
            return self.get_completion(prompt)
        except Exception as e:
            logger.error(f"Error generating initial summary: {str(e)}")
            raise
    
    def simplify_chunk(self, chunk: str, context: str) -> str:
        """Simplify a specific chunk of text while maintaining academic integrity."""
        try:
            grade_desc = self.get_grade_level_description()
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

            Overall Paper Context:
            {context}
            
            Chunk to Simplify:
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
                    f.write(f"### Section {i}\n\n")
                    
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

    def process_paper(self, input_pdf_path: str, output_json_path: str = None, output_text_path: str = None, output_pdf_path: str = None):
        """Main method to process and simplify an academic paper."""
        try:
            logger.info("Starting paper processing...")
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self.extract_text_from_pdf(input_pdf_path)
            
            # Generate initial summary
            logger.info("Generating initial summary...")
            summary = self.get_initial_summary(text)
            
            # Initialize output files if requested
            if output_text_path:
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    # Write initial markdown content
                    f.write("# Simplified Academic Paper\n\n")
                    f.write("> **Note**: This document contains mathematical equations in LaTeX format. ")
                    f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.\n\n")
                    f.write("## Summary\n\n")
                    # Process summary and write it
                    processed_summary = (summary.replace("\\(", "$")
                                            .replace("\\)", "$")
                                            .replace("\\[", "$$")
                                            .replace("\\]", "$$"))
                    f.write(f"{processed_summary}\n\n")
                    f.write("## Main Content\n\n")
            
            if output_json_path:
                # Initialize JSON structure
                json_content = {
                    "summary": summary,
                    "chunks": [],
                    "metadata": {
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                # Write initial JSON
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, ensure_ascii=False, indent=2)
            
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            chunks = self.split_text(text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process each chunk
            logger.info("Processing chunks...")
            simplified_chunks = [summary]  # Keep track of all chunks for PDF
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)}...")
                simplified_chunk = self.simplify_chunk(chunk, summary)
                simplified_chunks.append(simplified_chunk)
                
                # Update markdown file in real-time if requested
                if output_text_path:
                    with open(output_text_path, 'a', encoding='utf-8') as f:
                        f.write(f"### Section {i}\n\n")
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
                        f.write(f"{processed_chunk}\n\n")
                
                # Update JSON file in real-time if requested
                if output_json_path:
                    with open(output_json_path, 'r+', encoding='utf-8') as f:
                        data = json.load(f)
                        data["chunks"].append(simplified_chunk)
                        data["metadata"]["total_chunks"] = len(data["chunks"])
                        f.seek(0)
                        json.dump(data, f, ensure_ascii=False, indent=2)
                        f.truncate()
                
                # Respect rate limits
                time.sleep(float(os.getenv("MIN_TIME_BETWEEN_CALLS", "0.2")))
            
            # Add final markdown footer if needed
            if output_text_path:
                with open(output_text_path, 'a', encoding='utf-8') as f:
                    f.write("\n---\n\n")
                    f.write("*Note: This document uses KaTeX/MathJax compatible math notation. ")
                    f.write("For best viewing in VS Code, install the 'Markdown Preview Enhanced' extension.*\n")
            
            # Create PDF if requested (this still happens at the end due to PDF format requirements)
            if output_pdf_path:
                logger.info("Creating simplified PDF...")
                self.create_simplified_pdf(simplified_chunks, output_pdf_path)
                logger.info(f"Successfully saved simplified paper to {output_pdf_path}")
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Simplify academic papers while preserving mathematical content')
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('-g', '--grade-level', 
                       choices=['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6', 'grade7', 'grade8',
                               'grade9', 'grade10', 'grade11', 'grade12', 'freshman', 'undergraduate', 
                               'bachelors', 'masters', 'phd'],
                       default='undergraduate',
                       help='Target grade level for simplification (default: undergraduate)')
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
        
        # Get input file path and create output paths
        input_pdf = args.input
        base_name = os.path.splitext(input_pdf)[0]
        
        # Generate output filenames
        output_json = f"{base_name}_simplified_{args.grade_level}.json"
        output_text = f"{base_name}_simplified_{args.grade_level}.md"
        
        logger.info(f"Processing {input_pdf}")
        logger.info(f"Target grade level: {args.grade_level}")
        logger.info(f"Using AI provider: {args.provider}")
        logger.info(f"Output files will be:")
        logger.info(f"  - Markdown: {output_text}")
        logger.info(f"  - JSON: {output_json}")
        
        # Process paper and save to both JSON and text
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
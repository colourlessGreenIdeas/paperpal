#!/usr/bin/env python3

import os
import argparse
import logging
import markdown
from pathlib import Path
import tempfile
import pdfkit
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_tables(content):
    """
    Preprocess markdown tables to ensure they are properly formatted.
    """
    lines = content.split('\n')
    processed_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        # Check if this line might be part of a table
        if '|' in line:
            stripped = line.strip()
            if not in_table:
                # If this is a potential table start, add a blank line before if needed
                if processed_lines and processed_lines[-1].strip():
                    processed_lines.append('')
                in_table = True
            
            # Clean up the table row
            cells = [cell.strip() for cell in stripped.split('|')]
            # Remove empty cells from start/end
            if not cells[0]: cells = cells[1:]
            if not cells[-1]: cells = cells[:-1]
            
            # Format the row properly
            processed_line = '| ' + ' | '.join(cells) + ' |'
            
            # If this is a separator row (contains ---)
            if all(cell.replace('-', '').replace(':', '').strip() == '' for cell in cells):
                # Ensure each cell has at least 3 dashes
                cells = [':---:' if ':' in cell else '---' for cell in cells]
                processed_line = '| ' + ' | '.join(cells) + ' |'
            
            processed_lines.append(processed_line)
        else:
            if in_table and stripped:
                # Add blank line after table
                processed_lines.append('')
            in_table = False
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def convert_md_to_pdf(input_path: str, output_path: str = None):
    """
    Convert a Markdown file to PDF while preserving LaTeX equations and table formatting.
    
    Args:
        input_path (str): Path to the input Markdown file
        output_path (str, optional): Path for the output PDF file. If not provided,
                                   will use the same name as input with .pdf extension
    """
    try:
        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # If output path not provided, create one from input path
        if not output_path:
            output_path = str(Path(input_path).with_suffix('.pdf'))
            
        logger.info(f"Converting {input_path} to {output_path}")
        
        # Read markdown content
        with open(input_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Preprocess tables
        md_content = preprocess_tables(md_content)
        
        # Save preprocessed markdown for debugging
        # Uncomment to save the preprocessed markdown for debugging
        # debug_md_path = str(Path(input_path).with_suffix('.debug.md'))
        # with open(debug_md_path, 'w', encoding='utf-8') as f:
        #     f.write(md_content)
        # logger.info(f"Saved preprocessed markdown to {debug_md_path}")
        
        # Configure Markdown extensions
        md = markdown.Markdown(extensions=[
            'mdx_math',  # LaTeX math support
            'tables',    # Table support
            'fenced_code',  # Code block support
            'footnotes',    # Footnote support
            'attr_list',    # Attribute list support
            'def_list',     # Definition list support
            'abbr',         # Abbreviation support
            'meta'          # Metadata support
        ])
        
        # Convert to HTML
        html_content = md.convert(md_content)
        
        # Save intermediate HTML for debugging
        # Uncomment to save the preprocessed markdown for debugging
        # debug_html_path = str(Path(input_path).with_suffix('.debug.html'))
        # with open(debug_html_path, 'w', encoding='utf-8') as f:
        #     f.write(html_content)
        # logger.info(f"Saved intermediate HTML to {debug_html_path}")
        
        # Full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
            <script type="text/x-mathjax-config">
                MathJax.Hub.Config({{
                    tex2jax: {{
                        inlineMath: [['$','$']],
                        displayMath: [['$$','$$']],
                        processEscapes: true
                    }},
                    "HTML-CSS": {{ scale: 100 }}
                }});
            </script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    padding: 2em;
                    max-width: 50em;
                    margin: auto;
                }}
                h1, h2, h3 {{ 
                    color: #2c3e50;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                code {{ 
                    background: #f8f9fa;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                }}
                pre {{ 
                    background: #f8f9fa;
                    padding: 1em;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                blockquote {{
                    border-left: 4px solid #ccc;
                    margin: 1em 0;
                    padding-left: 1em;
                    color: #666;
                }}
                /* Table Styles */
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1.5em 0;
                    font-size: 0.9em;
                    min-width: 400px;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    color: #2c3e50;
                    font-weight: bold;
                    text-align: left;
                    padding: 12px 8px;
                    white-space: nowrap;
                }}
                td {{
                    padding: 10px 8px;
                    vertical-align: top;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .MathJax {{ 
                    font-size: 1.1em !important;
                }}
                /* Table Caption/Title */
                table caption {{
                    font-weight: bold;
                    text-align: left;
                    margin-bottom: 0.5em;
                    font-size: 1.1em;
                    color: #2c3e50;
                    padding: 0.5em 0;
                }}
            </style>
        </head>
        <body>
            {html_content}
            <script>
                // Wait for MathJax to finish rendering
                MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
            </script>
        </body>
        </html>
        """
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', encoding='utf-8', delete=False) as temp_html:
            temp_html.write(full_html)
            temp_html_path = temp_html.name
        
        try:
            # Configure pdfkit options
            options = {
                'enable-local-file-access': None,
                'javascript-delay': 2000,  # Wait 2 seconds for MathJax to render
                'no-stop-slow-scripts': None,
                'enable-javascript': None,
                'page-size': 'A4',
                'margin-top': '20mm',
                'margin-right': '20mm',
                'margin-bottom': '20mm',
                'margin-left': '20mm',
                'encoding': 'UTF-8',
                'quiet': None,
                'print-media-type': None,
                'enable-smart-shrinking': None
            }
            
            # Convert HTML to PDF using pdfkit
            pdfkit.from_file(temp_html_path, output_path, options=options)
            logger.info(f"Successfully converted {input_path} to {output_path}")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_html_path)
        
    except Exception as e:
        logger.error(f"Error converting markdown to PDF: {str(e)}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert Markdown file to PDF with LaTeX and table support')
    parser.add_argument('input', help='Input markdown file path')
    parser.add_argument('-o', '--output', help='Output PDF file path (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Convert the file
        convert_md_to_pdf(args.input, args.output)
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 
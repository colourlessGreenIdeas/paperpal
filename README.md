# Paperpal: Academic Paper Simplifier

A Python-based tool that simplifies research papers while preserving key insights. It aims to maintain consistency between original paper and the simplified one by providing section by section simplification. It processes PDF based papers, creates simplified versions, and outputs in markdown. Once output in Markdown, users can view it using a Markdown Preview feature like in VSCode or convert it into pdf using `md_to_pdf.py`

## Features

### Paper Simplification
- Extracts text from academic PDFs (although it could theoretically be used to simplify any content within any pdf document)
- Accepts as arguments education levels (Grade 1 through PhD) to dumb it down to that level
- Generates comprehensive summaries
- Breaks down papers into semantic chunks
- Preserves any mathematical equations and formulas
- Maintains paper structure and key insights
- Real-time output preview during processing

### Output Formats
- **Markdown**: Clean, readable format with LaTeX equation support
- **JSON**: Structured data format for further processing. *This output is turned off for now since I don't quite know what to do with it yet*
- **PDF**: to convert the markdown file into pdf, run the `md_to_pdf.py` script

## Prerequisites

### System Dependencies

#### Ubuntu/Debian
```bash
    python3.10 
    wkhtmltopdf
    poppler-utils
```
Install wkhtmltopdf and poppler-utils like so:

```bash
sudo apt-get install -y poppler-utils
sudo apt-get install -y wkhtmltopdf
```

#### Windows
1. Install Python 3.10 or higher from https://www.python.org/downloads/
2. Download and install wkhtmltopdf from https://wkhtmltopdf.org/downloads.html
   - Add wkhtmltopdf to your system's PATH (usually `C:\Program Files\wkhtmltopdf\bin`)
3. Install poppler for Windows:
   - Download from http://blog.alivate.com.au/poppler-windows/
   - Extract to a permanent location
   - Add poppler's bin directory to your system's PATH

### Python Environment Setup

You can use either `venv` or `conda` to manage your Python environment. Choose one of the following methods:

#### Option 1: Using Conda (recommended)

1. Install Miniconda or Anaconda if you haven't already:
   - Download from https://docs.conda.io/en/latest/miniconda.html

2. Create a new conda environment with Python 3.10:
```bash
# Create new environment
conda create -n paperagent python=3.10

# Activate the environment
conda activate paperagent

# Verify Python version
python --version  # Should show Python 3.10.x
```
#### Option 2: Using venv (Python's built-in virtual environment)

1. Create a new virtual environment with Python 3.10:
```bash
# Linux/macOS
python3.10 -m venv venv
source venv/bin/activate

# Windows
# First, ensure Python 3.10 is your default Python or use full path
python -m venv venv
.\venv\Scripts\activate
```

2. Verify Python version:
```bash
python --version  # Should show Python 3.10.x
```


### Python Dependencies

After setting up and activating your environment (either venv or conda), install the required packages:

```bash
pip install -r requirements.txt
```

The project uses several Python packages:

#### Core Dependencies
- `openai`: For OpenAI and Azure OpenAI API
- `PyPDF2`: PDF text extraction (v3.0.1)
- `tiktoken`: Token counting for API calls (v0.5.2)
- `langchain`: LLM framework and utilities
- `python-dotenv`: Environment variable management (v1.0.0)

#### PDF and Markdown Processing
- `pdfkit`: PDF generation (v1.0.0)
- `Markdown`: Markdown processing (v3.4.3)
- `python-markdown-math`: LaTeX math support (v0.8)
- `weasyprint`: Alternative PDF generation (v52.5)
- `Jinja2`: HTML templating (v3.0.3)

## OpenAI Requirements:
- Open AI Key
- Open AI Model name
- Open AI Embedding Model name

### Azure OpenAI Requirements
- Azure OpenAI API key
- Azure OpenAI endpoint
- Deployment name for GPT model

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Set up Python environment (choose one method):
```bash
# Method 1: venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Method 2: conda
conda create -n paperagent python=3.10
conda activate paperagent
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file. Copy the `.example.env` files contents to `.env` and edit it based on your requirements:

## Usage

### Basic Usage
```bash
python paperpal.py input.pdf
```

### Specify Education Level
You can target different education levels:
```bash
# Elementary School
python paperpal.py input.pdf -g grade1

# High School
python paperpal.py input.pdf -g grade10

# College Level (default)
python paperpal.py input.pdf -g undergraduate

# Graduate Level
python paperpal.py input.pdf -g phd
```

Available grade levels:
- Elementary: `grade1` through `grade6`
- Middle School: `grade7` and `grade8`
- High School: `grade9` through `grade12`
- College: `freshman`, `undergraduate`, `bachelors`
- Graduate: `masters`, `phd`

### Output Files
The script generates files with the grade level in the filename:
- `{input}_simplified_{grade_level}.md`: Markdown output
- `{input}_simplified_{grade_level}.json`: JSON output

### To convert the output markdown (.md) files to pdf, run the `md_to_pdf.py` script:

`python3 md_to_pdf.py bitcoin_simplified_grade12.md`

optionally, provide an outputname if you wish:

`python3 md_to_pdf.py bitcoin_simplified_grade12.md -o bitcoin_for_dummies.pdf`

## Troubleshooting

### Common Installation Issues

1. **wkhtmltopdf Issues**
   - Verify installation: `wkhtmltopdf --version`
   - Check PATH environment variable
   - Try reinstalling with system package manager

2. **poppler-utils Issues**
   - Verify installation: `pdfinfo -v`
   - Check if binaries are in PATH
   - Windows: Ensure proper path to poppler binaries

3. **Python Package Issues**
   ```bash
   # If you encounter SSL or package download issues
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   
   # If you need to upgrade pip
   python -m pip install --upgrade pip
   ```

4. **PDF Processing Issues**
   - Ensure PDF is not encrypted
   - Check file permissions
   - Verify poppler installation

### Environment Issues

1. **Azure OpenAI Configuration**
   - Verify all environment variables are set
   - Check API key format
   - Confirm endpoint URL format
   - Verify deployment name exists

2. **Virtual Environment**
   - Ensure venv is activated
   - Check Python version: `python --version`
   - Verify package installation: `pip list`

## Configuration

### Paper Simplifier Settings
Edit `paperpal.py` to customize:
- Chunk size and overlap
- Model temperature
- Output formatting preferences
- Rate limiting parameters

### PDF Converter Settings
The PDF conversion includes options for:
- Page size and margins
- Font styles and sizes
- Equation rendering delay
- Layout customization

## Troubleshooting

### Common Installation Issues

1. **wkhtmltopdf Issues**
   - Verify installation: `wkhtmltopdf --version`
   - Check PATH environment variable
   - Try reinstalling with system package manager

2. **poppler-utils Issues**
   - Verify installation: `pdfinfo -v`
   - Check if binaries are in PATH
   - Windows: Ensure proper path to poppler binaries

3. **Python Package Issues**
   ```bash
   # If you encounter SSL or package download issues
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   
   # If you need to upgrade pip
   python -m pip install --upgrade pip
   ```

4. **PDF Processing Issues**
   - Ensure PDF is not encrypted
   - Check file permissions
   - Verify poppler installation

### Environment Issues

1. **Azure OpenAI Configuration**
   - Verify all environment variables are set
   - Check API key format
   - Confirm endpoint URL format
   - Verify deployment name exists

2. **Virtual Environment**
   - Ensure venv is activated
   - Check Python version: `python --version`
   - Verify package installation: `pip list`

## Configuration

### Paper Simplifier Settings
Edit `paperpal.py` to customize:
- Chunk size and overlap
- Model temperature
- Output formatting preferences
- Rate limiting parameters

### PDF Converter Settings
The PDF conversion includes options for:
- Page size and margins
- Font styles and sizes
- Equation rendering delay
- Layout customization

## License

MIT

## Contributing

Contributions are welcome! Feel free to:
- Submit issues
- Fork the repository
- Create pull requests
- Suggest improvements

## Roadmap

- [ ] Performance improvements
  - Implement async/await for parallel processing
  - Add batch processing capabilities
  - Optimize the code
  - Add a recursive summary
  - Add image and figure support

- [ ] Additional Model Providers
  - Add Anthropic Claude support
  - Add Google Gemini support
  - Add local model support via Ollama/others?
  - Support custom model deployments

- [ ] Web Application
  - Develop web interface
  - Add user authentication
  - Real-time processing status
  - Collaborative features

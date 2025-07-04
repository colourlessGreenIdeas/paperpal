# Paperpal: Academic Paper Simplifier

Transform complex content into clear, understandable formats for any education level. Paperpal processes various types of content (PDFs, text, YouTube videos) and generates simplified versions while preserving key insights and mathematical formulas.

## 🚀 Quick Start

1. **Install Python 3.10+** if you haven't already
2. **Install in one command:**
   ```bash
   git clone <repository-url> && cd paperpal && pip install -e .
   ```
3. **Start the web app:**
   ```bash
   uvicorn app:app --reload
   ```
4. Open `http://localhost:8000/paperpal.html` in your browser
5. Upload content and choose your desired education level!

Need more details? Check the [detailed installation guide](#installation) for OS-specific prerequisites.

## ✨ Features

### Content Processing
- 📚 **Multiple Content Types**:
  - Academic PDFs and documents
  - Plain text articles
  - YouTube video transcripts
  - Web articles
- 🎓 **Education Level Targeting**:
  - Supports Grade 1 through PhD levels
  - Adjusts complexity while maintaining accuracy
  - Perfect for students, educators, and researchers
- 🧮 **Smart Processing**:
  - Breaks content into semantic chunks
  - Preserves mathematical equations and formulas
  - Maintains original structure and key insights
  - Real-time output preview during processing in markdown format
  - Maintains connections between simplified sections

### Output Formats
- 🌐 **Web Interface**: Easy-to-use browser-based interface with real-time processing
- 📝 **Markdown**: Clean, readable format with LaTeX equation support
- 📊 **JSON**: Structured data format for further processing
- 📄 **PDF Export**: Convert simplified content to PDF using `md_to_pdf.py`

### AI Providers
- 🤖 **Multiple Backends**:
  - OpenAI (GPT-4, GPT-3.5)
  - Azure OpenAI
  - Google Gemini
  - More coming soon!


## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)
- Git

<details>
<summary>Windows Prerequisites</summary>

1. Install Microsoft Visual C++ Redistributable:
   - Download and install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

2. After installing Python dependencies, you'll need to install browser binaries:
   ```bash
   playwright install
   ```
</details>

<details>
<summary>Linux Prerequisites</summary>

1. Install required system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y wget curl unzip libglib2.0-0 libnss3 libnspr4 \
       libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 \
       libxcb1 libxkbcommon0 libx11-6 libxcomposite1 libxdamage1 \
       libxext6 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
       libcairo2 libasound2 libatspi2.0-0 libwayland-client0

   # Fedora
   sudo dnf install -y alsa-lib atk at-spi2-atk at-spi2-core cairo cups-libs \
       dbus-libs expat GConf2 gdk-pixbuf2 glib2 gtk3 libdrm libX11 \
       libXcomposite libXdamage libXext libXfixes libxkbcommon \
       libXrandr libxshmfence nspr nss pango wget
   ```

2. After installing Python dependencies, you'll need to install browser binaries:
   ```bash
   playwright install
   ```
</details>

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd paperpal
   ```

2. Install the package (this will automatically install all dependencies including Playwright browsers):
   ```bash
   pip install -e .
   ```

That's it! The installation process will automatically handle all Python dependencies and Playwright browser installations.

> Note: If you prefer to install dependencies manually, you can still use:
> ```bash
> pip install -r requirements.txt
> playwright install
> ```

## 📄 Usage

### Web App (Recommended)
1. Start the server:
   ```bash
   uvicorn app:app --reload
   ```
2. Open `http://localhost:8000/paperpal.html` in your browser
3. Upload your academic paper (PDF)
4. Choose the target education level
5. Click "Simplify" and wait for the results

### Command Line Interface
```bash
python cli.py simplify --input paper.pdf --level "high_school" --output simplified.pdf
```

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

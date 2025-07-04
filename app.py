from fastapi import FastAPI, UploadFile, HTTPException, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
import json
from typing import List, Dict
import uuid
from paperpal import Paperpal, AsyncRateLimiter, CacheManager, OpenAIModel, GeminiModel, AzureOpenAIModel, LanguageModel
import asyncio
# from contentprocessor import PdfProcessor, TextProcessor, WebProcessor
from pydantic import BaseModel
import logging as logger
from webtopdf import webpage_to_pdf
from youtube_transcript import extract_youtube_transcript

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# Initialize PaperPal
rate_limiter = AsyncRateLimiter(rate_limit=50, per_seconds=60)
cache_manager = CacheManager()
model = GeminiModel(rate_limiter)
paper_pal = Paperpal(model, cache_manager, temp=0.3)

GRADE_LEVELS = ["grade4", "grade8", "grade12", "undergraduate", "phd"]

# Mount the output/images directory
app.mount("/images", StaticFiles(directory=os.path.join(OUTPUT_DIR, "images")), name="images")

class URLRequest(BaseModel):
    url: str

class RenameRequest(BaseModel):
    new_name: str

class TextUpload(BaseModel):
    text: str

class YouTubeRequest(BaseModel):
    url: str

# API routes
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"detail": "Only PDF files are allowed"}
            )
        
        # Generate unique ID for the content
        content_id = str(uuid.uuid4())
        
        # Save original PDF
        pdf_path = os.path.join(UPLOAD_DIR, f"{content_id}.pdf")
        
        # Read file content
        file_content = await file.read()
        
        # Save the file
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(file_content)
        
        try:
            # Process paper for all grade levels
            await paper_pal.process_paper(
                input_pdf=pdf_path,
                output_dir=OUTPUT_DIR,
                grade_levels=GRADE_LEVELS,
                content_id=content_id
            )
            
            # Create metadata
            metadata = {
                "content_id": content_id,
                "content_type": "pdf",
                "original_filename": file.filename,
                "versions": {grade: f"{content_id}_{grade}.json" for grade in GRADE_LEVELS}
            }
            
            metadata_path = os.path.join(OUTPUT_DIR, f"{content_id}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return JSONResponse(content=metadata)
            
        except Exception as e:
            # Clean up uploaded file if processing fails
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error processing PDF: {str(e)}"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error uploading file: {str(e)}"}
        )

@app.post("/api/upload/text")
async def upload_text(text: str = Body(..., embed=True)):
    try:
        # Generate unique ID for the content
        content_id = str(uuid.uuid4())
        
        # Save the text to a file
        text_path = os.path.join(UPLOAD_DIR, f"{content_id}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        try:
            # Process text for all grade levels
            await paper_pal.process_paper(
                input_pdf=text_path,  # We'll update this to handle text files
                output_dir=OUTPUT_DIR,
                grade_levels=GRADE_LEVELS,
                content_id=content_id
            )
            
            # Create metadata
            metadata = {
                "content_id": content_id,
                "content_type": "text",
                "original_filename": "text_input.txt",
                "versions": {grade: f"{content_id}_{grade}.json" for grade in GRADE_LEVELS}
            }
            
            metadata_path = os.path.join(OUTPUT_DIR, f"{content_id}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return JSONResponse(content=metadata)
            
        except Exception as e:
            # Clean up uploaded file if processing fails
            if os.path.exists(text_path):
                os.remove(text_path)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error processing text: {str(e)}"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error handling text: {str(e)}"}
        )

@app.get("/api/papers")
async def list_papers() -> List[Dict]:
    try:
        papers = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('_metadata.json'):
                with open(os.path.join(OUTPUT_DIR, filename)) as f:
                    metadata = json.load(f)
                    papers.append(metadata)
        return JSONResponse(content=papers)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error listing papers: {str(e)}"}
        )

@app.get("/api/paper/{content_id}/{grade_level}")
async def get_paper_version(content_id: str, grade_level: str):
    try:
        if grade_level not in GRADE_LEVELS:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid grade level"}
            )
        
        file_path = os.path.join(OUTPUT_DIR, f"{content_id}_{grade_level}.json")
        try:
            with open(file_path) as f:
                content = json.load(f)
                return JSONResponse(content=content)
        except FileNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Paper version not found"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving paper version: {str(e)}"}
        )

@app.post("/api/upload/url")
async def upload_url(request: URLRequest):
    try:
        # Generate unique ID for the content
        content_id = str(uuid.uuid4())

        # Save PDF in uploads directory
        pdf_path = os.path.join(UPLOAD_DIR, f"{content_id}.pdf")
        
        try:
            # Convert URL to PDF
            logger.info(f"Converting URL to PDF: {request.url}")
            webpage_to_pdf(request.url, pdf_path) #This saves the pdf in the uploads directory
            
            if not os.path.exists(pdf_path):
                raise RuntimeError("PDF conversion failed")
            
            # Process paper for all grade levels
            await paper_pal.process_paper(
                input_pdf=pdf_path,
                output_dir=OUTPUT_DIR,
                grade_levels=GRADE_LEVELS,
                content_id=content_id
            )
            
            # Create metadata
            metadata = {
                "content_id": content_id,
                "content_type": "web",
                "original_url": request.url,
                "versions": {grade: f"{content_id}_{grade}.json" for grade in GRADE_LEVELS}
            }
            
            metadata_path = os.path.join(OUTPUT_DIR, f"{content_id}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return JSONResponse(content=metadata)
            
        except Exception as e:
            # Clean up PDF if conversion or processing fails
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except:
                    pass
            logger.error(f"Error processing URL content: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error processing URL content: {str(e)}"}
            )
            
    except Exception as e:
        logger.error(f"Error handling URL upload: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error handling URL upload: {str(e)}"}
        )

@app.post("/api/paper/{content_id}/rename")
async def rename_paper(content_id: str, request: RenameRequest):
    try:
        metadata_path = os.path.join(OUTPUT_DIR, f"{content_id}_metadata.json")
        if not os.path.exists(metadata_path):
            return JSONResponse(
                status_code=404,
                content={"detail": "Paper not found"}
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata["original_filename"] = request.new_name

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return JSONResponse(content={"message": "Paper renamed successfully"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error renaming paper: {str(e)}"}
        )

@app.delete("/api/paper/{content_id}")
async def delete_paper(content_id: str):
    try:
        metadata_path = os.path.join(OUTPUT_DIR, f"{content_id}_metadata.json")
        if not os.path.exists(metadata_path):
            return JSONResponse(
                status_code=404,
                content={"detail": "Paper not found"}
            )

        # Read metadata to get all associated files
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Delete all version files
        for version_file in metadata.get("versions", {}).values():
            version_path = os.path.join(OUTPUT_DIR, version_file)
            if os.path.exists(version_path):
                os.remove(version_path)

            # Delete markdown file as well
            markdown_path = os.path.join(OUTPUT_DIR, version_file.replace(".json", ".md"))
            if os.path.exists(markdown_path):
                os.remove(markdown_path)

        # Delete original files
        original_files = [
            os.path.join(UPLOAD_DIR, f"{content_id}.pdf"),
            os.path.join(UPLOAD_DIR, f"{content_id}.txt"),
            metadata_path
        ]

        for file_path in original_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete any associated images
        image_dir = os.path.join(OUTPUT_DIR, "images")
        for file in os.listdir(image_dir):
            if file.startswith(content_id):
                os.remove(os.path.join(image_dir, file))

        return JSONResponse(content={"message": "Paper deleted successfully"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error deleting paper: {str(e)}"}
        )

@app.post("/api/youtube")
async def process_youtube(request: YouTubeRequest):
    """Process YouTube video transcript"""
    try:
        # Extract transcript
        transcript = extract_youtube_transcript(request.url)
        if transcript.startswith("Error") or transcript == "Invalid YouTube URL":
            raise HTTPException(status_code=400, detail=transcript)

        return await upload_text(transcript)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files last
app.mount("/", StaticFiles(directory=".", html=True), name="static") 
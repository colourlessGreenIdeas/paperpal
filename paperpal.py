import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import tiktoken
import time
import logging
import json
from typing import List, Dict, Optional, Union, Protocol
import argparse 
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiofiles
import hashlib
from tqdm.asyncio import tqdm_asyncio
import re
from contentprocessor import PdfProcessor, TextProcessor
import uuid

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
logger.info(".env file loaded.")

# --- System Prompt ---
SYSTEM_PROMPT = """You are an expert academic researcher who excels at ensuring accuracy of complex content and simplifying it.
When handling mathematical equations:
1. Use $ for inline equations (e.g., $x = y$).
2. Use $$ for display equations (e.g., $$\\sum_{i=1}^n x_i$$).
3. Never use \\( or \\) or \\[ or \\] for equations.
4. Use proper LaTeX notation for mathematical symbols.
5. Keep variable names and mathematical symbols consistent.
6. Do not add any additional text, commentary, or your own thoughts outside the simplified content.
7. **IMPORTANT**: If you see placeholders for images and figures in the input, do not incldue them in your output
"""

# --- Grade Level Descriptions ---
GRADE_LEVEL_DESCRIPTIONS = {
    "original": "return the original text without any simplification",
    "grade1": "a 6-7 year old first grader, using very simple words and concepts",
    "grade4": "a 9-10 year old fourth grader, using more advanced vocabulary",
    "grade8": "a 13-14 year old eighth grader, using algebra-level math concepts",
    "undergraduate": "a general undergraduate student with basic field knowledge",
    "phd": "a PhD level researcher with expert field knowledge"
    # Add more as needed
}

# --- Async Rate Limiter ---
class AsyncRateLimiter:
    def __init__(self, rate_limit: int = 50, per_seconds: int = 60):
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.allowance = rate_limit
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self.lock:
            now = time.monotonic()
            time_passed = now - self.last_check
            self.last_check = now
            self.allowance += time_passed * (self.rate_limit / self.per_seconds)

            if self.allowance > self.rate_limit:
                self.allowance = self.rate_limit

            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per_seconds / self.rate_limit)
                logger.debug(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
                self.allowance = 1.0 # Reset after sleeping

            self.allowance -= 1.0

# --- Language Model Abstraction ---
class LanguageModel(Protocol):
    async def get_completion(self, prompt: str, system_message: str, temperature: float) -> str:
        ...

# --- OpenAI Implementation ---
from openai import AsyncOpenAI

class OpenAIModel(LanguageModel):
    def __init__(self, rate_limiter: AsyncRateLimiter):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.rate_limiter = rate_limiter

    async def get_completion(self, prompt: str, system_message: str, temperature: float) -> str:
        await self.rate_limiter.wait_if_needed()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            return f"ERROR: OpenAI failed - {e}"

# --- Gemini Implementation ---
from google import genai
from google.genai import types

class GeminiModel(LanguageModel):
    def __init__(self, rate_limiter: AsyncRateLimiter):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not set")

        # --- Use the new genai.Client ---
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        self.rate_limiter = rate_limiter
        logger.info(f"Initialized Google GenAI SDK (google-genai) for model: {self.model_name}")

    def _run_inference(self, prompt: str, system_message: str, temperature: float) -> str:
        """Synchronous method using your specific generate_content call."""
        try:
            # --- This part comes directly from your provided code ---
            response = self.client.models.generate_content(
                model=self.model_name,  # Use the model name from init
                contents=prompt,       # Pass the user prompt as contents
                config=types.GenerateContentConfig(
                    temperature=temperature,  # Use the passed temperature
                    system_instruction=system_message # Pass the system message
                )
            )
            # --- End of your provided code ---
            return response.text
        except Exception as e:
            logger.error(f"Gemini API Error (google-genai): {e}")
            # Add more details if available
            if hasattr(e, 'response') and e.response:
                 logger.error(f"Gemini Response Details: {e.response}")
            return f"ERROR: Gemini (google-genai) failed - {e}"

    async def get_completion(self, prompt: str, system_message: str, temperature: float) -> str:
        """Asynchronously calls the synchronous inference method."""
        await self.rate_limiter.wait_if_needed()
        # --- Run the synchronous call in a separate thread ---
        return await asyncio.to_thread(self._run_inference, prompt, system_message, temperature)


# --- Azure OpenAI Implementation ---
from openai import AsyncAzureOpenAI

class AzureOpenAIModel(LanguageModel):
    def __init__(self, rate_limiter: AsyncRateLimiter):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        if not self.deployment_name: raise ValueError("AZURE_DEPLOYMENT_NAME not set")
        self.rate_limiter = rate_limiter

    async def get_completion(self, prompt: str, system_message: str, temperature: float) -> str:
        await self.rate_limiter.wait_if_needed()
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Azure OpenAI API Error: {e}")
            return f"ERROR: Azure OpenAI failed - {e}"


# --- Hugging Face Implementation (Sync wrapped in Async) ---
# from transformers import pipeline as hf_pipeline
# import torch

# class HuggingFaceModel(LanguageModel):
#     _pipeline = None # Class-level pipeline to avoid reloading

#     def __init__(self, model_name: str = "google/flan-t5-large"):
#         self.model_name = model_name
#         if HuggingFaceModel._pipeline is None:
#             try:
#                 device = 0 if torch.cuda.is_available() else -1
#                 HuggingFaceModel._pipeline = hf_pipeline(
#                     "text2text-generation", model=self.model_name, device=device
#                 )
#                 logger.info(f"Initialized Hugging Face pipeline for '{self.model_name}' on {'GPU' if device == 0 else 'CPU'}")
#             except Exception as e:
#                 logger.error(f"Failed to load HF model '{self.model_name}': {e}")
#                 raise

#     def _run_inference(self, prompt: str, system_message: str, temperature: float) -> str:
#         # Note: Temp & system_message handling is basic for HF pipelines.
#         # It's better to fine-tune models or use chat-specific ones.
#         full_prompt = f"{system_message}\n\nTask:\n{prompt}"
#         try:
#             # Adjust params as needed. temperature isn't always a direct param.
#             response = HuggingFaceModel._pipeline(full_prompt, max_length=1500, num_return_sequences=1)
#             return response[0]['generated_text']
#         except Exception as e:
#             logger.error(f"Hugging Face Inference Error: {e}")
#             return f"ERROR: Hugging Face failed - {e}"

#     async def get_completion(self, prompt: str, system_message: str, temperature: float) -> str:
#         # Run the synchronous pipeline in a separate thread
#         return await asyncio.to_thread(self._run_inference, prompt, system_message, temperature)


# --- Cache Manager ---
class CacheManager:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, *args) -> str:
        s = "".join(map(str, args)).encode('utf-8')
        return hashlib.md5(s).hexdigest()

    async def get(self, chunk: str, context: str, grade_level: str) -> Optional[str]:
        key = self._get_cache_key(chunk, context, grade_level)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                    data = await f.read()
                    logger.debug(f"Cache HIT for key {key}")
                    return json.loads(data)["content"]
            except Exception as e:
                logger.warning(f"Cache read error for {key}: {e}")
                return None
        logger.debug(f"Cache MISS for key {key}")
        return None

    async def set(self, chunk: str, context: str, grade_level: str, content: str):
        key = self._get_cache_key(chunk, context, grade_level)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps({"content": content}))
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")

# --- Paperpal Core Class ---
class Paperpal:
    def __init__(self, model: LanguageModel, cache_manager: CacheManager, temp: float = 0.3):
        self.model = model
        self.cache_manager = cache_manager
        self.temp = temp
        self.chunk_size = 1024
        self.chunk_overlap = 100
        
        # Initialize processors
        self.pdf_processor = PdfProcessor()
        self.text_processor = TextProcessor()
        # self.web_processor = WebToPdfProcessor()

    def get_grade_level_description(self, grade_level: str) -> str:
        return GRADE_LEVEL_DESCRIPTIONS.get(grade_level, "an undergraduate student")

    def get_prompt(self, chunk: str, context: str, grade_level: str) -> str:
        grade_desc = self.get_grade_level_description(grade_level)
        if grade_level == "original":
            return f"Return the following text exactly as provided, preserving all formatting and placeholders like [IMAGE:...]:\n\n{chunk}"
        else:
            return f"""Using the overall context and the current chunk, create a simplified version suitable for {grade_desc}.
            Follow these rules:
            1. Use language appropriate for {grade_desc}.
            2. Explain complex concepts simply.
            3. if equations are present, either simplify them orpreserve them exactly as they appear in the input
            4. Preserve important information and equations (adjusting complexity).
            5. Do not add commentary. Focus only on simplification.
            6. **IMPORTANT**: If you see placeholders for images and figures in the input, do not incldue them in your output

            Overall Paper Context:
            {context}

            Current Chunk to Simplify:
            {chunk}

            Simplified version:"""

    async def simplify_chunk(self, chunk: str, context: str, grade_level: str) -> str:
        cached = await self.cache_manager.get(chunk, context, grade_level)
        if cached:
            return cached

        prompt = self.get_prompt(chunk, context, grade_level)
        simplified = await self.model.get_completion(prompt, SYSTEM_PROMPT, self.temp)

        await self.cache_manager.set(chunk, context, grade_level, simplified)
        return simplified

    async def process_all_grade_levels(self, chunks: List[Dict], grade_levels: List[str]) -> Dict[str, List[str]]:
        results = {grade: [] for grade in grade_levels}
        llm_tasks = []
        llm_task_indices = []
        # Track which images have already been inserted
        inserted_images = {grade: set() for grade in grade_levels}

        # Schedule LLM calls for all text chunks
        for idx, chunk in enumerate(chunks):
            for grade_level in grade_levels:
                if chunk['text'].strip():
                    context = ''  # You can add context logic if needed
                    prompt = self.get_prompt(chunk['text'], context, grade_level)
                    llm_tasks.append(self.model.get_completion(prompt, SYSTEM_PROMPT, self.temp))
                    llm_task_indices.append((idx, grade_level))
                else:
                    llm_tasks.append(None)
                    llm_task_indices.append((idx, grade_level))

        # Run all LLM calls in parallel (filter out None)
        llm_results = await asyncio.gather(*[t for t in llm_tasks if t is not None])
        llm_iter = iter(llm_results)
        llm_outputs = {}
        for i, t in enumerate(llm_tasks):
            idx, grade_level = llm_task_indices[i]
            if t is not None:
                llm_outputs[(idx, grade_level)] = next(llm_iter)
            else:
                llm_outputs[(idx, grade_level)] = ''

        # Now, assemble the final output, appending images only after the first chunk in which they appear
        for grade_level in grade_levels:
            seen_images = set()
            for idx, chunk in enumerate(chunks):
                text = llm_outputs[(idx, grade_level)]
                results[grade_level].append(text)
                for img in chunk['images']:
                    if img not in seen_images:
                        results[grade_level].append(img)
                        seen_images.add(img)
        return results

    def _format_content(self, content: str) -> str:
        formatted = (content.replace("\\(", "$").replace("\\)", "$")
                           .replace("\\[", "$$").replace("\\]", "$$"))
        return formatted

    async def _save_markdown(self, chunks: List[str], output_path: Path, grade_level: str, image_dir: str):
        output_dir = output_path.parent
        image_rel_dir = Path(image_dir).relative_to(output_dir)

        def replace_image_tag(match):
            img_filename = match.group(1)
            img_rel_path = (image_rel_dir / img_filename).as_posix() # Use forward slashes
            return f"![{img_filename}]({img_rel_path})"

        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(f"# Simplified Academic Paper - {grade_level}\n\n")
            await f.write("> **Note**: Check image paths and ensure LaTeX renders correctly.\n\n")
            for chunk in chunks:
                formatted = self._format_content(chunk)
                processed = re.sub(r"\[IMAGE:\s*(.*?)\s*\]", replace_image_tag, formatted)
                await f.write(f"{processed}\n\n")
        logger.info(f"Saved {grade_level} markdown to {output_path}")

    async def _save_json(self, chunks: List[str], output_path: Path, grade_level: str):
         formatted_chunks = [self._format_content(chunk) for chunk in chunks]
         content = {
            "metadata": {
                "grade_level": grade_level, "total_chunks": len(chunks),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "chunks": formatted_chunks
         }
         async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
             await f.write(json.dumps(content, ensure_ascii=False, indent=2))
         logger.info(f"Saved {grade_level} JSON to {output_path}")

    async def process_paper(self, input_pdf: str, output_dir: str, grade_levels: List[str], content_id: str = None, content_type: str = None):
        """
        Process a paper and generate simplified versions for different grade levels.

        :param input_pdf: Path to the input file (PDF or text) or URL string
        :param output_dir: Directory to save output files
        :param grade_levels: List of grade levels to generate
        :param content_id: Unique identifier for this content (optional)
        :param content_type: Type of content ('pdf', 'text', or 'web')
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        image_dir = output_path / "images"
        image_dir.mkdir(exist_ok=True)

        if not content_id:
            content_id = str(uuid.uuid4())

        # Choose processor based on content type or file extension
        if content_type == 'web':
            # processor = self.web_processor
            # source = input_pdf  # For web, this is the URL
            raise ValueError("Web processing not implemented yet")
        else:
            input_path = Path(input_pdf)
            source = str(input_path)
            processor = self.text_processor if input_path.suffix.lower() == '.txt' else self.pdf_processor
        
        content = processor.extract_content(
            source,
            output_image_dir=str(image_dir),
            content_id=content_id
        )
        chunks = processor.create_chunks(
            content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        if not chunks:
            logger.error("No text chunks could be created. Aborting.")
            return

        simplified_content = await self.process_all_grade_levels(chunks, grade_levels)

        save_tasks = []
        for grade_level, grade_chunks in simplified_content.items():
            md_path = output_path / f"{content_id}_{grade_level}.md"
            json_path = output_path / f"{content_id}_{grade_level}.json"
            save_tasks.append(self._save_markdown(grade_chunks, md_path, grade_level, str(image_dir)))
            save_tasks.append(self._save_json(grade_chunks, json_path, grade_level))

        await asyncio.gather(*save_tasks)
        logger.info(f"Processing complete! Output saved to {output_dir}")

# --- Main Execution ---
async def main():
    parser = argparse.ArgumentParser(description='Simplify academic papers.')
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('-o', '--output-dir', default='output', help='Output directory')
    parser.add_argument('-g', '--grade-levels', nargs='+',
                        choices=list(GRADE_LEVEL_DESCRIPTIONS.keys()),
                        default=['grade4', 'grade8', 'grade12', 'undergraduate', 'phd'],
                        help='Target grade levels')
    parser.add_argument('-p', '--provider', choices=['openai', 'gemini', 'azure', 'hf'],
                        default='gemini', help='AI provider to use')
    parser.add_argument('--hf-model', default='google/flan-t5-large',
                        help='Hugging Face model name if provider is hf')
    parser.add_argument('--rate-limit', type=int, default=50,
                        help='API calls per minute (approximate)')
    parser.add_argument('--temp', type=float, default=0.3, help='LLM Temperature')
    args = parser.parse_args()

    rate_limiter = AsyncRateLimiter(rate_limit=args.rate_limit)
    cache_manager = CacheManager()

    model: LanguageModel
    if args.provider == 'openai':
        model = OpenAIModel(rate_limiter)
    elif args.provider == 'gemini':
        model = GeminiModel(rate_limiter)
    elif args.provider == 'azure':
        model = AzureOpenAIModel(rate_limiter)
    # elif args.provider == 'hf':
    #     model = HuggingFaceModel(args.hf_model)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    simplifier = Paperpal(model, cache_manager, temp=args.temp)
    await simplifier.process_paper(args.input, args.output_dir, args.grade_levels)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
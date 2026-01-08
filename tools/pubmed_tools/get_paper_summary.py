# Suppress Google GenAI warnings
import warnings
import logging
import os

# Suppress specific Google GenAI warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google_genai")
warnings.filterwarnings("ignore", message=".*GOOGLE_API_KEY.*GEMINI_API_KEY.*")
warnings.filterwarnings("ignore", message=".*AFC is enabled.*")

# Set logging levels to ERROR for google_genai modules
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)

# Suppress HTTPX logging messages
logging.getLogger("httpx").setLevel(logging.ERROR)

from google import genai
from google.genai import types
import pathlib
import httpx
import asyncio
import aiofiles
import json
import re
from dotenv import load_dotenv
load_dotenv("../.env")

def extract_json_from_response(response_text: str) -> str:
    """
    Extract and clean JSON from LLM response, handling markdown code blocks and syntax errors.
    """
    if not response_text:
        return response_text
    
    # First, try to extract JSON from markdown code blocks
    patterns = [
        r'```json\s*\n(.*?)\n```',  # Standard markdown JSON block
        r'```json\s*(.*?)```',      # JSON block without newlines
        r'```\s*\n(.*?)\n```',      # Generic code block
        r'```\s*(.*?)```'           # Generic code block without newlines
    ]
    
    json_content = None
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            json_content = matches[0].strip()
            break
    
    # If no markdown blocks found, assume the entire response is JSON
    if json_content is None:
        json_content = response_text.strip()
    
    # Try to parse and validate the JSON
    try:
        # First, try parsing as-is
        parsed = json.loads(json_content)
        return json.dumps(parsed)  # Return properly formatted JSON
    except json.JSONDecodeError:
        # If parsing fails, try to fix common syntax issues
        try:
            fixed_json = fix_json_syntax_issues(json_content)
            parsed = json.loads(fixed_json)
            return json.dumps(parsed)  # Return properly formatted JSON
        except json.JSONDecodeError as e:
            print(f"[Warning] Could not parse JSON from LLM response: {e}")
            # Return the original response as a fallback
            return response_text

def fix_json_syntax_issues(json_str: str) -> str:
    """
    Fix common JSON syntax issues like missing commas.
    """
    # Split into lines and fix missing commas
    lines = json_str.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # Skip empty lines and structural lines
        if not line.strip() or line.strip() in ['{', '}']:
            fixed_lines.append(line)
            continue
        
        # Check if we need to add a comma
        if i < len(lines) - 1:
            # Look for the next non-empty line
            next_non_empty = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and lines[j].strip() not in ['{', '}']:
                    next_non_empty = lines[j].strip()
                    break
            
            # If current line ends with a value and next line starts a new field, add comma
            if (next_non_empty and 
                next_non_empty.startswith('"') and 
                ':' in next_non_empty and 
                not line.endswith(',') and 
                not line.endswith('{') and 
                not line.endswith('[')):
                
                # Add comma if the line ends with a value
                if (line.strip().endswith('"') or 
                    line.strip().endswith('}') or 
                    line.strip().endswith(']') or
                    re.search(r'(true|false|null|\d+)$', line.strip())):
                    line += ','
        
        fixed_lines.append(line)
    
    # Remove trailing commas before closing braces
    result = '\n'.join(fixed_lines)
    result = re.sub(r',(\s*})', r'\1', result)
    result = re.sub(r',(\s*])', r'\1', result)
    
    return result

def has_error_content(text: str) -> bool:
    """
    Check if the text contains error messages indicating failed paper download/parsing.
    """
    error_indicators = [
        "[Error] : No result can be found",
        "No result can be found",
        "<B> - https://www.ncbi.nlm.nih.gov/",
        "Functions: Extracting pubmed abstracts"
    ]
    
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    text_lower = text.lower()
    for indicator in error_indicators:
        if indicator.lower() in text_lower:
            return True
    return False

import asyncio
import time
from collections import deque

class RateLimiter:
    """Rate limiter for LLM API calls"""
    
    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make an API call"""
        async with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            while self.calls and self.calls[0] < now - 60:
                self.calls.popleft()
            
            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls_per_minute:
                sleep_time = 60 - (now - self.calls[0]) + 0.1  # Small buffer
                print(f"[Rate Limit] Waiting {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                return await self.acquire()  # Recursive call after waiting
            
            # Record this call
            self.calls.append(now)

# Global rate limiter instance
_rate_limiter = RateLimiter(max_calls_per_minute=50)  # Conservative limit

async def get_paper_summary_async(file_path: str = None, text: str = None, mine_type: str = None) -> str:
    """
    Async version of get_paper_summary using LLM, but first check if the content contains error messages.
    Returns None if the paper contains error content.
    """
    # Apply rate limiting for LLM calls
    await _rate_limiter.acquire()
    
    client = genai.Client()

    if mine_type == "application/pdf":
        filepath = pathlib.Path(file_path)
        if not filepath.exists() or filepath.stat().st_size == 0:
            return None
        # Read file asynchronously
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
    elif text:
        # Check if text contains error messages
        if has_error_content(text):
            print(f"[Debug] Detected error content in text, skipping LLM processing")
            return None
        mine_type = "text/plain"
        if isinstance(text, str):
            data = text.encode('utf-8')
        else:
            data = text
    else:
        return None

    try:
        prompt = """
        For each paper, you must thoroughly analyze the complete text content provided.

        Do not rely solely on abstracts or metadata if the full text is available. Your synthesis needs to be based on a comprehensive understanding of the entire paper.

        Information Extraction and Synthesis: From the full text of each paper, extract and synthesize the following specific components. Ensure the information is relevant to the original search keyword/phrase you generated. Your descriptions for each component should be detailed, fact-based, and directly supported by the paper's text.

        Objective/Purpose:

        Clearly and comprehensively state all specific primary and secondary questions, aims, or hypotheses the study sought to address, as explicitly mentioned by the authors.

        Quote or closely paraphrase the research questions if they are clearly articulated.

        Avoid inferring objectives not explicitly stated.

        This section should be at least 100 words

        Methodology:

        Provide a detailed account of the study's design (e.g., Randomized Controlled Trial, Meta-Analysis, Cohort Study, Case-Control, etc.), including specific phases if applicable.

        Describe the key methods and procedures used for data collection in detail.

        Detail participant characteristics: inclusion and exclusion criteria, demographics, and the final sample size for each group if applicable.

        Specify the main outcome measures and how they were assessed/measured.

        Outline the data analysis techniques and statistical tests employed.

        For intervention studies, clearly describe the intervention(s) and control group conditions.

        This section should be at least 200 words

        Key Findings:

        Present the most important results and data points comprehensively and factually.

        For each primary and secondary outcome, report the specific findings, including exact figures, percentages, and units where applicable.

        Crucially, include significant quantitative data such as p-values, confidence intervals (CIs), effect sizes (e.g., odds ratios, relative risks, Cohen's d), and other relevant statistical measures reported by the authors.

        Clearly link findings back to the study's objectives and outcome measures. Distinguish between statistically significant and non-significant results.

        Report findings related to different subgroups if analyzed and reported by the authors.

        This section should be at least 100 words

        Authors' Conclusions:

        Detail the main conclusions drawn by the authors in their own words or a very close paraphrase.

        Explain how the authors interpreted their findings in the context of their research questions and the existing literature.

        Include any implications of the findings for practice, policy, or future research as suggested by the authors.

        This should be more than a restatement of the key findings; it's about the authors' interpretation and takeaways.

        This section should be at least 100 words

        Relevance to Search Query:

        Provide a detailed explanation of how the paper's specific findings, methodology, or conclusions directly address or contribute to understanding the initial search keyword/phrase.

        Analyze whether the paper confirms, refutes, expands upon, or adds nuance to existing knowledge related to the query topic.

        Be specific in connecting elements of the paper to the query.

        This section should be at least 50 words

        IMPORTANT: You MUST return your response as a valid JSON object with the following exact structure:
        
        {
          "objective": "The extracted objective/purpose as a string",
          "methodology": "The extracted methodology as a string",
          "key_findings": "The extracted key findings as a string",
          "authors_conclusions": "The extracted authors' conclusions as a string",
          "relevance_to_search_query": "The determined relevance to the search query as a string",
          "limitations_noted": "The extracted limitations as a string",
          "analysis_status": "A brief status, e.g., 'Analysis complete' or 'Error: Insufficient text for analysis.'"
        }
        
        Do NOT wrap the JSON in markdown code blocks or any other formatting. Return only the raw JSON object.
        """

        # Run the LLM call in a thread pool to avoid blocking
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=data,
                        mime_type=mine_type,
                    ),
                    prompt
                ]
            )
        )
        
        # Check if the LLM response also contains error indicators
        if has_error_content(response.text):
            print(f"[Debug] LLM response contains error content, filtering out")
            return None
        
        # Extract and clean JSON from the response
        cleaned_response = extract_json_from_response(response.text)
        return cleaned_response
    except Exception as e:
        print(f"[Error] Failed to process with LLM: {e}")
        return None

def get_paper_summary(file_path: str = None, text: str = None, mine_type: str = None) -> str:
    """
    Synchronous wrapper for backward compatibility.
    """
    return asyncio.run(get_paper_summary_async(file_path, text, mine_type))
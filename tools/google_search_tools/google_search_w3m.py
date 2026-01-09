import os
import sys
import time
from unittest import result
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import json
load_dotenv()

import subprocess
import argparse
import asyncio

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.path_config import get_path

# Add OpenAI API import
try:
    import openai
except ImportError:
    openai = None

# Add AutoGen imports for agent mode (optional, only used for AutoGen agent mode)
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import StructuredMessage
    from autogen_agentchat.ui import Console
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Silent import - warning only shown when AutoGen features are actually needed


def google_search(query: str, target_results: int = 10, use_llm_filter: bool = True) -> list:  # type: ignore[type-arg]

    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://www.googleapis.com/customsearch/v1"
    
    all_results = []
    seen_urls = set()  # Track URLs to avoid duplicates
    
    # Start with more results to account for LLM filtering
    initial_search_count = target_results * 2 if use_llm_filter else target_results
    max_search_attempts = min(100, initial_search_count)  # Google API limit
    
    # Calculate number of API calls needed (max 10 results per call)
    calls_needed = min((max_search_attempts + 9) // 10, 10)  # Google limits to 100 total results
    
    print(f"Target: {target_results} results. Searching {max_search_attempts} initially...")
    print(f"Making {calls_needed} API calls...")
    
    # First round of searches
    for call_num in range(calls_needed):
        start_index = call_num * 10 + 1  # Google uses 1-based indexing
        
        params = {
            "key": api_key, 
            "cx": search_engine_id, 
            "q": query, 
            "num": 10,
            "start": start_index
        }

        try:
            response = requests.get(url, params=params)  # type: ignore[arg-type]

            if response.status_code != 200:
                print(f"API call {call_num + 1} failed: {response.status_code}")
                print(response.json())
                continue

            batch_results = response.json().get("items", [])
            
            # Filter out duplicate URLs
            unique_batch_results = []
            for item in batch_results:
                if item["link"] not in seen_urls:
                    seen_urls.add(item["link"])
                    unique_batch_results.append(item)
                else:
                    print(f"Skipping duplicate: {item['link']}")
            
            all_results.extend(unique_batch_results)
            print(f"API call {call_num + 1}: Got {len(unique_batch_results)} new results (total: {len(all_results)})")
            
            # Add a small delay between API calls to be respectful
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in API call {call_num + 1}: {str(e)}")
            continue

    print(f"Initial search complete: {len(all_results)} unique URLs collected")
    results = all_results

    def get_page_content(url: str) -> str:
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Check if the content is likely binary (PDF, etc.)
            content_type = response.headers.get('content-type', '').lower()
            if any(binary_type in content_type for binary_type in ['pdf', 'application/octet-stream', 'application/msword']):
                return f"[Binary content detected: {content_type}] Unable to extract text from binary file."
            
            # Try to decode with proper encoding detection
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            
            # Return full content without any truncation - leverage Gemini's 1M context
            return text.strip()
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching {url}: {str(e)}")
            return ""
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    def process_search_result(item):
        """Process a single search result with w3m"""
        url = item['link']
        
        # Check if URL points to a PDF or other binary file
        binary_extensions = ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
                            '.zip', '.rar', '.tar', '.gz', '.exe', '.dmg', '.pkg',
                            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
                            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv']
        
        # Handle PDFs separately - try to extract text but with fallback
        if url.lower().endswith('.pdf'):
            print(f"Processing PDF file: {url}")
            try:
                # Try w3m first for PDF text extraction
                result = subprocess.run(
                    ["w3m", "-dump", "-cols", "120", url],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=30
                )
                cleaned_text = result.stdout
                
                # If w3m fails or returns minimal content for PDF, use snippet
                if len(cleaned_text.strip()) < 100:
                    print(f"PDF text extraction minimal, using snippet for {url}")
                    cleaned_text = f"[PDF Document] {item.get('snippet', '')} [Note: PDF text extraction was limited]"
                else:
                    print(f"Successfully extracted {len(cleaned_text)} characters from PDF")
                    
            except Exception as e:
                print(f"PDF processing failed for {url}: {e}")
                cleaned_text = f"[PDF Document] {item.get('snippet', '')} [Note: PDF text extraction failed]"
            
            return {
                "title": item["title"], 
                "link": item["link"], 
                "snippet": item["snippet"], 
                "body": cleaned_text
            }
        
        # Skip other binary files entirely
        if any(url.lower().endswith(ext) for ext in binary_extensions):
            file_type = url.split('.')[-1].upper()
            print(f"Skipping binary file: {url}")
            return {
                "title": item["title"], 
                "link": item["link"], 
                "snippet": item["snippet"], 
                "body": f"[Binary file: {file_type}] {item.get('snippet', '')}"
            }
        
        try:
            # Use w3m to get clean text content with proper encoding handling
            result = subprocess.run(
                ["w3m", "-dump", "-cols", "120", url],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',  # Ignore encoding errors instead of failing
                timeout=30
            )
            cleaned_text = result.stdout
            
            # Check for w3m encoding issues
            if "Some characters could not be decoded" in cleaned_text or "REPLACEMENT CHARACTER" in cleaned_text:
                print(f"w3m encoding issues detected for {url}, trying fallback method")
                cleaned_text = get_page_content(url)
            elif len(cleaned_text.strip()) < 50:
                print(f"w3m returned minimal content for {url}, trying fallback method")
                cleaned_text = get_page_content(url)
            
            # Check for excessive replacement characters (���)
            replacement_char_count = cleaned_text.count('�')
            if replacement_char_count > 10:  # Arbitrary threshold for "too many" replacement characters
                print(f"w3m produced {replacement_char_count} replacement characters for {url}, trying fallback method")
                cleaned_text = get_page_content(url)
                
        except subprocess.TimeoutExpired:
            print(f"Timeout fetching {url}")
            cleaned_text = ""
        except UnicodeDecodeError as e:
            print(f"Encoding error fetching {url} with w3m: {e}")
            # Fallback to the original method for encoding issues
            cleaned_text = get_page_content(url)
        except Exception as e:
            print(f"Error fetching {url} with w3m: {str(e)}")
            # Fallback to the original method
            cleaned_text = get_page_content(url)

        return {
            "title": item["title"], 
            "link": item["link"], 
            "snippet": item["snippet"], 
            "body": cleaned_text
        }

    enriched_results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:  # Increased workers for more results
        # Submit all tasks
        future_to_item = {executor.submit(process_search_result, item): item for item in results}
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            try:
                result = future.result()
                enriched_results.append(result)
            except Exception as e:
                item = future_to_item[future]
                print(f"Error processing {item['link']}: {str(e)}")
                # Add a minimal result for failed items
                enriched_results.append({
                    "title": item["title"], 
                    "link": item["link"], 
                    "snippet": item["snippet"], 
                    "body": ""
                })

    print(f"Content extraction complete: {len(enriched_results)} results")

    # Apply LLM filtering and analysis if requested
    if use_llm_filter:
        print("Starting LLM analysis and filtering...")
        filtered_results, rejected_results = process_results_with_llm(enriched_results, query, target_results)
        all_rejected = rejected_results.copy()  # Keep track of all rejected results
        
        # If we don't have enough results after filtering, search for more
        attempts = 0
        max_attempts = 3
        
        while len(filtered_results) < target_results and attempts < max_attempts:
            attempts += 1
            shortage = target_results - len(filtered_results)
            print(f"Attempt {attempts}: Need {shortage} more results. Searching additional pages...")
            
            # Search more pages
            additional_start = len(all_results) + 1
            additional_calls = min(3, (shortage * 2 + 9) // 10)  # Search 2x the shortage
            
            for call_num in range(additional_calls):
                start_index = additional_start + (call_num * 10)
                if start_index > 100:  # Google API limit
                    break
                    
                params = {
                    "key": api_key, 
                    "cx": search_engine_id, 
                    "q": query, 
                    "num": 10,
                    "start": start_index
                }

                try:
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        batch_results = response.json().get("items", [])
                        new_results = []
                        
                        for item in batch_results:
                            if item["link"] not in seen_urls:
                                seen_urls.add(item["link"])
                                new_results.append(item)
                        
                        if new_results:
                            # Process new results
                            with ThreadPoolExecutor(max_workers=10) as executor:
                                new_futures = {executor.submit(process_search_result, item): item for item in new_results}
                                new_enriched = []
                                
                                for future in as_completed(new_futures):
                                    try:
                                        result = future.result()
                                        new_enriched.append(result)
                                    except Exception as e:
                                        print(f"Error processing additional result: {e}")
                            
                            # Filter new results with LLM
                            if new_enriched:
                                additional_filtered, additional_rejected = process_results_with_llm(new_enriched, query, shortage)
                                filtered_results.extend(additional_filtered)
                                all_rejected.extend(additional_rejected)
                                print(f"Added {len(additional_filtered)} more results (total: {len(filtered_results)})")
                            
                            if len(filtered_results) >= target_results:
                                break
                    
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error in additional search: {e}")
            
            if len(filtered_results) >= target_results:
                break
        
        final_results = filtered_results[:target_results]
        print(f"Final result: {len(final_results)} LLM-filtered results")
        
        # Create a results container with rejected results for analysis
        class SearchResults(list):
            def __init__(self, results, rejected_results):
                super().__init__(results)
                self.rejected_results = rejected_results
        
        return SearchResults(final_results, all_rejected)
    
    else:
        # Return original results without LLM filtering
        return enriched_results[:target_results]


def setup_openai():
    """Initialize OpenAI API"""
    if openai is None:
        raise ImportError("openai not installed. Install with: pip install openai")
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=api_key)
    return client


def analyze_content_with_llm(content: str, title: str, url: str, query: str, client, max_retries: int = 3) -> dict:
    """
    Use OpenAI GPT-4o-mini to filter, analyze and summarize webpage content with retry logic for API failures
    
    Returns:
        dict with 'is_relevant', 'summary', 'confidence' keys
    """
    # Check content quality first
    content_length = len(content.strip())
    
    # Truncate very long content to manage token usage
    max_content_length = 8000  # Roughly 2000 tokens, leaving room for prompt
    if content_length > max_content_length:
        content = content[:max_content_length] + "\n\n[Content truncated for analysis]"
        print(f"Truncated content from {content_length} to {len(content)} characters for analysis")
    
    # Filter out low-quality content
    if content_length < 200:
        return {
            'is_relevant': False,
            'summary': '',
            'confidence': 0.0,
            'reason': f'Content too short ({content_length} characters)'
        }
    
    # Check for common scraping failure indicators - be very specific to avoid false positives
    failure_indicators = [
        'page not found',
        '404 error',
        '404 - not found',
        'access denied',
        'permission denied',
        'please enable cookies to continue',
        'bot protection active',
        'captcha verification',
        'cloudflare protection',
        'this site is temporarily unavailable'
    ]
    
    content_lower = content.lower()
    for indicator in failure_indicators:
        if indicator in content_lower:
            return {
                'is_relevant': False,
                'summary': '',
                'confidence': 0.0,
                'reason': f'Scraping failure detected: {indicator}'
            }
    
    # Create analysis prompt
    prompt = f"""
You are a research analyst tasked with evaluating webpage content for relevance to a specific search query and creating comprehensive summaries.

SEARCH QUERY: "{query}"
WEBPAGE TITLE: "{title}"
WEBPAGE URL: {url}

WEBPAGE CONTENT:
{content}

CONTENT ANALYSIS INSTRUCTIONS:

1. RELEVANCE ASSESSMENT:
   - Determine if the content has ANY direct or indirect connection to the search query
   - Be generous in relevance assessment - include content that is tangentially related
   - Only exclude content that is completely unrelated, broken, or purely navigational
   - Navigation elements like "Skip to main content" are normal and don't disqualify a page

2. SUMMARY REQUIREMENTS (if relevant):
   - Write 8-12 sentences that directly reflect the search query
   - Include specific factual information: names, dates, locations, numbers, percentages
   - Focus on key findings, conclusions, and evidence that answers the query
   - Begin directly with the information - DO NOT use introductory phrases like "This article/webpage/papers/ directly address..."
   - Maintain objectivity and use clear, precise language

3. KEY FINDINGS:
   - Extract 3-5 specific, actionable insights that directly relate to the query
   - Focus on concrete facts, discoveries, or conclusions
   - Include quantitative data where available

4. QUALITY INDICATORS TO INCLUDE:
   - Research findings and scientific studies
   - Expert opinions and recommendations  
   - Statistical data and evidence
   - Technical specifications or methodologies
   - Case studies and real-world applications

5. EXCLUDE ONLY IF:
   - Content is completely unrelated to the query topic
   - Page contains only error messages or broken content
   - Content is purely navigational menus/headers/footers

RESPONSE FORMAT (JSON):
{{
    "is_relevant": true/false,
    "summary": "Comprehensive 8-12 sentence summary that directly addresses the query with specific facts, figures, and findings. Leave empty if not relevant.",
    "confidence": 0.0-1.0,
    "key_findings": ["specific finding 1", "specific finding 2", "specific finding 3"]
}}

EXAMPLE OUTPUT:
{{
    "is_relevant": true,
    "summary": "Current limitations of Large Language Models (LLMs) in Retrieval Augmented Generation (RAG) systems include several critical challenges. RAG systems exhibit brittleness, showing high sensitivity to minor changes in input queries, retrieval processes, or model parameters, which can significantly impact output quality. The retrieval component often struggles with semantic matching, frequently returning documents that are lexically similar but contextually irrelevant to the query. LLMs within RAG frameworks suffer from hallucination issues, where they generate plausible-sounding but factually incorrect information even when provided with accurate retrieved documents. Context window limitations restrict the amount of retrieved information that can be processed simultaneously, forcing systems to truncate potentially relevant content. The ranking and selection of retrieved documents remains problematic, as current methods may prioritize less relevant but more recent or popular content over highly relevant but older sources. Integration challenges arise when combining information from multiple retrieved documents, often resulting in contradictory or incoherent responses. Additionally, RAG systems lack robust evaluation metrics and standardized benchmarks, making it difficult to assess and compare system performance across different domains and use cases.",
    "confidence": 0.95,
    "key_findings": [
        "RAG systems exhibit brittleness with high sensitivity to input variations and parameter changes",
        "Semantic matching in retrieval often returns lexically similar but contextually irrelevant documents", 
        "LLMs continue to hallucinate even when provided with accurate retrieved information",
        "Context window limitations force truncation of potentially relevant retrieved content"
    ]
}}

Respond with valid JSON only:
"""
    
    # Retry logic for API failures
    for attempt in range(max_retries):
        try:
            # Use OpenAI API call
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are a research analyst tasked with evaluating webpage content for relevance to search queries and creating comprehensive summaries. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
          
            )
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
                
            result = json.loads(response_text)
            
            # Validate response structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
                
            required_keys = ['is_relevant', 'summary', 'confidence']
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing required key: {key}")
            
            # Ensure reason field exists for debugging
            if 'reason' not in result:
                if not result.get('is_relevant', False):
                    result['reason'] = 'LLM determined content not relevant'
                else:
                    result['reason'] = 'Content approved by LLM'
            
            return result
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"JSON parsing error for {url} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Raw response: {response_text[:500]}...")  # Show first 500 chars of response
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                print(f"JSON parsing error for {url} (final attempt): {e}")
                print(f"Raw response: {response_text[:500]}...")
                return {
                    'is_relevant': False,
                    'summary': '',
                    'confidence': 0.0,
                    'reason': f'LLM response parsing failed after {max_retries} attempts: {str(e)}'
                }
        except Exception as e:
            # Check if this is a retryable error (API/network issues)
            error_str = str(e).lower()
            retryable_errors = [
                'timeout', 'connection', 'network', 'rate limit', 'quota', 
                'service unavailable', '429', '500', '502', '503', '504',
                'temporary failure', 'try again', 'overloaded', 'resource_exhausted'
            ]
            
            is_retryable = any(error_type in error_str for error_type in retryable_errors)
            
            if is_retryable and attempt < max_retries - 1:
                # For rate limit errors (429), use longer delays
                if '429' in error_str or 'quota' in error_str or 'resource_exhausted' in error_str:
                    delay = 60 * (attempt + 1)  # 60s, 120s, 180s for rate limits
                    print(f"Rate limit hit for {url} (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Waiting {delay} seconds before retry...")
                else:
                    delay = 2 * (attempt + 1)  # Standard exponential backoff
                    print(f"Retryable API error for {url} (attempt {attempt + 1}/{max_retries}): {e}")
                
                time.sleep(delay)
                continue
            else:
                if is_retryable:
                    print(f"API error for {url} (final attempt): {e}")
                    return {
                        'is_relevant': False,
                        'summary': '',
                        'confidence': 0.0,
                        'reason': f'API failure after {max_retries} attempts: {str(e)}'
                    }
                else:
                    print(f"Non-retryable error for {url}: {e}")
                    return {
                        'is_relevant': False,
                        'summary': '',
                        'confidence': 0.0,
                        'reason': f'LLM analysis failed: {str(e)}'
                    }
    
    # This should never be reached, but just in case
    return {
        'is_relevant': False,
        'summary': '',
        'confidence': 0.0,
        'reason': f'Max retries ({max_retries}) exceeded'
    }


def process_results_with_llm(results: list, query: str, target_count: int = 40) -> tuple:
    """
    Process search results with LLM analysis and ensure target count is met
    
    Returns:
        tuple: (accepted_results, rejected_results) for filtering analysis
    """
    if openai is None:
        print("Warning: OpenAI not available, returning original results")
        return results[:target_count], []
    
    print(f"Analyzing {len(results)} results with LLM...")
    
    analyzed_results = []
    rejected_results = []
    
    # Process results with ThreadPoolExecutor for parallel LLM calls
    def analyze_single_result(result):
        try:
            # Add small delay to help with rate limiting
            time.sleep(0.5)  # 500ms delay between LLM calls
            
            # Create a new client for each thread to ensure thread safety
            client = setup_openai()
            
            print(f"Analyzing: {result['title'][:50]}... ({len(result['body']):,} chars)")
            
            analysis = analyze_content_with_llm(
                content=result['body'],
                title=result['title'],
                url=result['link'],
                query=query,
                client=client
            )
            
            # Check if analysis failed but didn't raise exception
            if analysis.get('reason', '').startswith('LLM analysis failed') or analysis.get('reason', '').startswith('LLM response parsing failed'):
                print(f"⚠️  LLM analysis issue for {result['link']}: {analysis.get('reason', 'Unknown error')}")
            
            # Always add LLM analysis data to the result
            result_with_analysis = {
                **result,
                'llm_summary': analysis['summary'],
                'llm_confidence': analysis['confidence'],
                'llm_key_findings': analysis.get('key_findings', []),
                'llm_is_relevant': analysis['is_relevant'],
                'llm_reason': analysis.get('reason', ''),
                'is_llm_analyzed': True
            }
            
            if analysis['is_relevant'] and analysis['confidence'] > 0.2:  # Lowered threshold from 0.3 to 0.2
                result_with_analysis['is_llm_filtered'] = True
                return ('accepted', result_with_analysis)
            else:
                result_with_analysis['is_llm_filtered'] = False
                rejection_reason = analysis.get('reason', 'Not relevant')
                content_preview = result['body'][:200].replace('\n', ' ')
                print(f"Filtered out: {result['link']}")
                print(f"  Reason: {rejection_reason} (confidence: {analysis['confidence']:.2f})")
                print(f"  Content length: {len(result['body']):,} chars")
                print(f"  Content preview: {content_preview}...")
                print(f"  Title: {result['title']}")
                return ('rejected', result_with_analysis)
        except Exception as e:
            print(f"Error in analyze_single_result for {result['link']}: {e}")
            error_result = {
                **result,
                'llm_summary': '',
                'llm_confidence': 0.0,
                'llm_key_findings': [],
                'llm_is_relevant': False,
                'llm_reason': f'Analysis failed: {str(e)}',
                'is_llm_analyzed': False,
                'is_llm_filtered': False
            }
            return ('rejected', error_result)
    
    # Parallel processing of LLM analysis with proper thread safety
    with ThreadPoolExecutor(max_workers=10) as executor:  # Increased to 10 for better API rate limiting
        future_to_result = {executor.submit(analyze_single_result, result): result for result in results}
        
        for future in as_completed(future_to_result):
            try:
                status, analyzed_result = future.result()
                if status == 'accepted':
                    analyzed_results.append(analyzed_result)
                    print(f"✓ Approved: {analyzed_result['title']} (confidence: {analyzed_result['llm_confidence']:.2f})")
                else:
                    rejected_results.append(analyzed_result)
            except Exception as e:
                original_result = future_to_result[future]
                print(f"✗ Analysis failed for {original_result['link']}: {e}")
                # Add to rejected with error info
                error_result = {
                    **original_result,
                    'llm_summary': '',
                    'llm_confidence': 0.0,
                    'llm_key_findings': [],
                    'llm_is_relevant': False,
                    'llm_reason': f'Processing failed: {str(e)}',
                    'is_llm_analyzed': False,
                    'is_llm_filtered': False
                }
                rejected_results.append(error_result)
    
    print(f"LLM Analysis complete: {len(analyzed_results)} accepted, {len(rejected_results)} rejected from {len(results)} total")
    
    # If we don't have enough results, we need to fetch more
    if len(analyzed_results) < target_count:
        shortage = target_count - len(analyzed_results)
        print(f"Need {shortage} more results. Current implementation returns what we have.")
        print(f"To get more results, increase the initial search count or implement additional search rounds.")
    
    return analyzed_results[:target_count], rejected_results



import time

def save_results_to_json(results, query, filename=None):
    """Save search results to a JSON file with minimal structure"""
    if filename is None:
        # Create filename based on query and timestamp
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')[:50]  # Limit length
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logs_dir = get_path('logs.base', absolute=True, create=True)
        filename = os.path.join(logs_dir, f"search_results_{safe_query}_{timestamp}.json")
    
    # Prepare data for JSON - simplified structure
    data = {
        "query": query,
        "results": []
    }
    
    # Process accepted results - simplified structure
    for i, result in enumerate(results, 1):
        result_data = {
            "index": i,
            "title": result["title"],
            "url": result["link"]
        }
        
        # Add LLM analysis data if available
        if 'llm_summary' in result:
            result_data.update({
                "llm_summary": result["llm_summary"],
                "llm_confidence": result["llm_confidence"],
                "llm_key_findings": result.get("llm_key_findings", [])
            })
        
        data["results"].append(result_data)
    
    # Save to JSON file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")
        return None

async def web_search_tool(query: str, target_results: int = 10, use_llm_filter: bool = True) -> str:
    """
    AutoGen tool wrapper for the Google search functionality.
    Performs web search and returns LLM-filtered JSON data for agent analysis.
    
    Args:
        query: Search query string
        target_results: Number of results to return (default: 10)
        use_llm_filter: Whether to use LLM filtering (default: True)
    
    Returns:
        JSON string containing curated search results with LLM analysis
    """
    try:
        print(f"🔍 Searching for: {query}")
        print(f"🎯 Target: {target_results} results with LLM filtering: {use_llm_filter}")
        
        # Perform the search
        results = google_search(query, target_results=target_results, use_llm_filter=use_llm_filter)
        
        if not results:
            return json.dumps({
                "query": query,
                "status": "no_results",
                "message": f"No results found for query: {query}",
                "results": []
            })
        
        # Prepare curated JSON data for agent analysis
        curated_data = {
            "query": query,
            "status": "success",
            "total_results": len(results),
            "results": []
        }
        
        for i, result in enumerate(results, 1):
            result_data = {
                "index": i,
                "title": result['title'],
                "url": result['link'],
                "is_relevant": result.get('llm_is_relevant', True),
                "summary": result.get('llm_summary', ''),
                "confidence": result.get('llm_confidence', 0.0),
                "key_findings": result.get('llm_key_findings', [])
            }
            curated_data["results"].append(result_data)
        
        # Save results to JSON file
        json_filename = save_results_to_json(results, query)
        if json_filename:
            curated_data["saved_file"] = json_filename
        
        # return json.dumps(curated_data, indent=2)
        return curated_data["results"]
        
    except Exception as e:
        error_msg = f"Error performing web search: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({
            "query": query,
            "status": "error",
            "message": error_msg,
            "results": []
        })

async def run_agent_mode(query: str, target_results: int = 10):
    """
    Run the search in AutoGen agent mode.
    
    Args:
        query: Search query
        target_results: Number of results to return
    """
    if not AUTOGEN_AVAILABLE:
        print("❌ AutoGen is not available. Please install it with:")
        print("   pip install autogen-agentchat autogen-ext[openai]")
        return
    
    
    try:
        # Create the model client
        model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-pro",  # Use Gemini 2.5 Pro for advanced capabilities
            model_info=ModelInfo(
                vision=True, 
                function_calling=True, 
                json_output=True, 
                family="GEMINI_2_5_PRO", 
                structured_output=False
            ),
        )
        
        # Create the web search tool function with proper signature
        async def web_search(query: str) -> str:
            """Search the web for information and return curated JSON data with LLM-analyzed summaries and key findings."""
            return await web_search_tool(query, target_results=target_results, use_llm_filter=True)
        
        # Create the assistant agent
        agent = AssistantAgent(
            name="research_assistant",
            model_client=model_client,
            tools=[web_search],
            reflect_on_tool_use=True,
            system_message="""You are an expert research analyst and report writer specializing in comprehensive information synthesis. Your task is to analyze curated web search results and produce detailed, well-structured research reports.

TASK OVERVIEW:
When you receive a research query, use the web_search tool to obtain LLM-analyzed summaries and key findings from multiple sources. Your goal is to synthesize this tool output into a comprehensive report of 1000+ words.

JSON DATA FORMAT YOU WILL RECEIVE:
The web_search tool returns JSON with this structure:
- query: The search query
- total_results: Number of curated results
- results: Array of objects, each containing:
  - title: Source title
  - url: Source URL
  - is_relevant: Boolean relevance flag
  - summary: 8-12 sentence LLM-generated summary
  - confidence: Relevance confidence score (0.0-1.0)
  - key_findings: Array of specific insights

REPORT STRUCTURE AND REQUIREMENTS:

1. **EXECUTIVE SUMMARY** (200-300 words)
   - Provide a high-level overview of the research topic
   - Highlight the most critical findings and trends
   - Summarize key challenges, opportunities, or implications

2. **DETAILED ANALYSIS** (1000-1500 words)
   - Synthesize information from all relevant sources
   - Organize content into logical themes or categories
   - Present evidence-based insights with specific facts and figures
   - Identify patterns, contradictions, or gaps in the information
   - Include quantitative data, statistics, and expert opinions where available


ANALYSIS GUIDELINES:

- **Source Integration**: Seamlessly weave information from multiple sources rather than summarizing each source separately
- **Critical Thinking**: Analyze relationships between different pieces of information, identify trends, and draw meaningful conclusions
- **Evidence-Based**: Support all claims with specific facts, statistics, or expert opinions from the sources
- **Balanced Perspective**: Present multiple viewpoints when they exist, noting areas of consensus and disagreement
- **Professional Tone**: Use clear, authoritative language appropriate for academic or business contexts
- **Source Attribution**: Reference sources by title and URL when making specific claims
- **Confidence Weighting**: Give more emphasis to findings from sources with higher confidence scores

OUTPUT FORMAT:
- Use clear headings for each section
- Include bullet points for lists and key findings
- Provide proper citations in the format: [Source Title](URL)
- Ensure the report flows logically and reads as a cohesive document
- Aim for 1500-2000 words total

QUALITY STANDARDS:
- Demonstrate deep understanding of the topic through synthesis rather than simple aggregation
- Provide actionable insights that go beyond surface-level observations
- Maintain objectivity while highlighting the most significant findings
- Ensure all factual claims are supported by source material
- Create a report that would be valuable for decision-makers or researchers in the field

Remember: Your role is not just to summarize but to analyze, synthesize, and create new insights from the curated information provided by the web search tool.""",
        )
        
        print(f"🤖 Starting AutoGen Research Assistant...")
        print(f"🔍 Query: {query}")
        print("=" * 60)
        
        # Run the agent with console output
        await Console(
            agent.run_stream(task=f"Research this topic and provide comprehensive insights: {query}"),
            output_stats=True,
            # stream_output=True,
        )
        
    except Exception as e:
        print(f"❌ Error in agent mode: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Google Search with LLM Analysis Tool")
    parser.add_argument("query", nargs="?", default="What are the key dysfunctional signaling targets in microglia of AD?", 
                       help="Search query (default: 'What is the current limitation of LLMs in RAG?')")
    parser.add_argument("--target", type=int, default=10, 
                       help="Target number of results (default: 10)")
    parser.add_argument("--no-filter", action="store_true", 
                       help="Disable LLM filtering")
    parser.add_argument("--agent", action="store_true", 
                       help="Run in AutoGen agent mode")
    
    args = parser.parse_args()
    
    query = args.query
    target_results = args.target
    use_llm_filtering = not args.no_filter
    
    if args.agent:
        # Run in agent mode
        asyncio.run(run_agent_mode(query, target_results))
    else:
        # Run in normal mode
        print(f"Searching for: {query}")
        print(f"Target: {target_results} high-quality results")
        print("Using parallel processing with w3m + GPT-4o-mini filtering...")
        
        t0 = time.time()
        results = google_search(query, target_results=target_results, use_llm_filter=use_llm_filtering)
        t1 = time.time()
        print(f"Search completed in {t1 - t0:.2f} seconds")
        print(f"Found {len(results)} high-quality results")
          
        # Print summary of results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['link']}")
            
            if 'llm_summary' in result:
                print(f"   LLM Confidence: {result['llm_confidence']:.2f}")
                print(f"   Content length: {len(result['body'])} characters")
                print(f"   Summary: {result['llm_summary'][:150]}...")
            else:
                print(f"   Content length: {len(result['body'])} characters")
        
        # Save results to JSON file
        json_filename = save_results_to_json(results, query)
        
        # Print statistics
        if use_llm_filtering:
            llm_filtered_count = sum(1 for r in results if 'llm_summary' in r)
            print(f"\nStatistics:")
            print(f"- LLM-filtered results: {llm_filtered_count}/{len(results)}")
            print(f"- Average confidence: {sum(r.get('llm_confidence', 0) for r in results) / len(results):.2f}")
        
        # Uncomment to see full results in console
        # print(results)
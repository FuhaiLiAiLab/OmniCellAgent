import os
from typing import Dict, List, Optional
from urllib.parse import urljoin
import html2text
import httpx
from autogen_core.code_executor import ImportFromModule
from autogen_core.tools import FunctionTool
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import asyncio
import concurrent.futures
from scrapegraphai.graphs import SmartScraperGraph

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def web_scraper_extract_sync(url: str, query: str) -> str:
    """
    Synchronous version of web scraper extract to avoid event loop conflicts.
    
    Args:
        url: The URL to scrape
        query: The search query to focus the extraction on
        
    Returns:
        Extracted information as a string
    """
    try:
        llm_instance_config = {
            "model": "gemini-2.5-flash",
            "google_api_key": os.getenv("GEMINI_API_KEY"),
        }

        llm_model_instance = ChatGoogleGenerativeAI(**llm_instance_config)

        # Configure the scraper with query-specific prompt
        graph_config = {
            "llm": {
                "model_instance": llm_model_instance,
                "model_tokens": 100000,
                "temperature": 0,
            },
            "verbose": False,
            "headless": True,
        }
        
        # Create a prompt that focuses on the search query




        # prompt = f"""
        # Extract information from this webpage that is relevant to the search query: "{query}"

        # Focus on:

        # #1 The main arguments, findings, or points that directly address the query.
        # #2 Key facts, statistics, or data points that provide evidence or support.
        # #3 Relevant background or context necessary to understand the main points.
        # #4 Any actionable insights, conclusions, or expert recommendations.

        # Instructions:

        # Exclude navigation menus, advertisements, unrelated sidebars, and any irrelevant content.
        # Avoid redundant details and general statements that do not contribute directly to the query.
        # Write a concise, comprehensive summary (8 to 10 sentences) that captures the essence of the page as it relates to the query, using clear and objective language.
        # """
        
        prompt = f"""
        Instructions:

        Exclude navigation menus, advertisements, unrelated sidebars, and any irrelevant content.
        Avoid redundant details and general statements that do not contribute directly to the query.
        Write a concise, comprehensive summary (8 to 10 sentences) that captures the essence of the page as it relates to the query, using clear and objective language.
        """

        smart_scraper = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=graph_config
        )
        
        result = smart_scraper.run()
            
        return result['content']
        
    except Exception as e:
        raise Exception(f"Error extracting content from {url}: {str(e)}")


async def web_scraper_extract(url: str, query: str) -> str:
    """
    Async wrapper for web scraper extract that runs in a thread executor.
    
    Args:
        url: The URL to scrape
        query: The search query to focus the extraction on
        
    Returns:
        Extracted information as a string
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            web_scraper_extract_sync, 
            url, 
            query
        )
        return result


async def fetch_google_search_results(
    query: str,
    num_results: int,
    start_index: int = 1,
    language: str = "en",
    country: Optional[str] = None,
    safe_search: bool = True,
) -> List[Dict[str, str]]:
    """
    Fetch search results from Google Custom Search API.
    
    Args:
        query: Search query string
        num_results: Number of results to fetch
        start_index: Starting index for results (1-based)
        language: Language code for search results
        country: Optional country code for search results
        safe_search: Enable safe search filtering
        
    Returns:
        List of search result items
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not cse_id:
        raise ValueError("Missing required environment variables. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID.")

    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": min(num_results, 10),  # Google API max is 10 per request
        "start": start_index,
        "hl": language,
        "safe": "active" if safe_search else "off",
    }

    if country:
        params["gl"] = country

    async with httpx.AsyncClient() as client:
        response = await client.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return data.get("items", [])


async def google_search_with_scraper(
    query: str,
    num_results: int = 40,
    include_snippets: bool = True,
    use_web_scraper: bool = True,
    content_max_length: Optional[int] = 10000,
    language: str = "en",
    country: Optional[str] = None,
    safe_search: bool = True,
    concurrent_scraping: bool = True,
    max_concurrent_scrapers: int = 10,
    max_retry_attempts: int = 50,  # Maximum total URLs to try
) -> List[Dict[str, str]]:
    """
    Perform a Google search using the Custom Search API and use web scraper to extract relevant content.
    Automatically retries with additional search results if extractions fail.
    Supports searching more than 10 results by making multiple API calls with pagination.

    Args:
        query: Search query string
        num_results: Number of successful results to return (no limit, automatically handles pagination)
        include_snippets: Include result snippets in output
        use_web_scraper: Use SmartScraperGraph to extract query-relevant content
        content_max_length: Maximum length of webpage content (fallback if scraper fails)
        language: Language code for search results (e.g., en, es, fr)
        country: Optional country code for search results (e.g., us, uk)
        safe_search: Enable safe search filtering
        concurrent_scraping: Enable concurrent processing of multiple URLs
        max_concurrent_scrapers: Maximum number of concurrent scraper tasks
        max_retry_attempts: Maximum total URLs to try before giving up

    Returns:
        List[Dict[str, str]]: List of successful search results
    """
    
    async def fetch_page_content(url: str, max_length: Optional[int] = 50000) -> str:
        """Helper function to fetch and convert webpage content to markdown (fallback)"""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Convert relative URLs to absolute
                for tag in soup.find_all(["a", "img"]):
                    if tag.get("href"):
                        tag["href"] = urljoin(url, tag["href"])
                    if tag.get("src"):
                        tag["src"] = urljoin(url, tag["src"])

                h2t = html2text.HTML2Text()
                h2t.body_width = 0
                h2t.ignore_images = False
                h2t.ignore_emphasis = False
                h2t.ignore_links = False
                h2t.ignore_tables = False

                markdown = h2t.handle(str(soup))

                return markdown.strip()

        except Exception as e:
            raise Exception(f"Error fetching content from {url}: {str(e)}")

    async def extract_full_title(url: str) -> str:
        """Extract the full title from the webpage's <title> tag"""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                title_tag = soup.find("title")
                
                if title_tag and title_tag.string:
                    return title_tag.string.strip()
                else:
                    return ""
        except Exception:
            return ""

    async def process_single_result(item: dict, processed_urls: set) -> Optional[dict]:
        """Process a single search result item with content extraction"""
        url = item.get("link", "")
        
        # Skip if we've already processed this URL
        if url in processed_urls:
            return None
            
        processed_urls.add(url)
        
        # Start with the Google API title (might be truncated)
        google_title = item.get("title", "")
        result = {"title": google_title, "link": url}
        
        if include_snippets:
            result["snippet"] = item.get("snippet", "")

        try:

            full_title = await extract_full_title(url)
            
            if use_web_scraper and (not str(url).endswith(".pdf")):
                print(f"Extracting content from: {url}")
                
                if full_title:
                    result["title"] = full_title
                    print(f"Updated title: {full_title}")
                
                extracted_content = await web_scraper_extract(url, query)
                result["content"] = extracted_content
                
                # Also include raw content as fallback
                result["raw_content"] = await fetch_page_content(url, max_length=content_max_length)
            else:
                
                if full_title:
                    result["title"] = full_title
                    
                result["content"] = await fetch_page_content(url, max_length=content_max_length)

            print(type(result["content"]))
             
            # assert isinstance(result["content"], str), "Content extraction failed, expected string"
            # assert len(result["content"]) > 0, "Content extraction returned empty string"
            assert result["content"] != "NA", "Content extraction returned 'NA', indicating failure"
            result["result_type"] = str(type(result["content"]))
            print(f"✓ Successfully extracted content from: {url}")
            return result
            
        except Exception as e:
            print(f"✗ Failed to extract content from {url}: {str(e)}")
            return None

    async def fetch_multiple_search_batches(
        query: str,
        total_needed: int,
        processed_urls: set,
        language: str = "en",
        country: Optional[str] = None,
        safe_search: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Fetch multiple batches of search results to get the required number of unique URLs.
        """
        all_items = []
        start_index = 1
        max_api_calls = (total_needed // 10) + 3  # Extra calls to account for duplicates
        
        for batch_num in range(max_api_calls):
            try:
                print(f"Fetching search batch {batch_num + 1}, results {start_index}-{start_index + 9}")
                
                batch_items = await fetch_google_search_results(
                    query=query,
                    num_results=10,  # Always fetch 10 (max per API call)
                    start_index=start_index,
                    language=language,
                    country=country,
                    safe_search=safe_search
                )
                
                if not batch_items:
                    print(f"No more search results available at index {start_index}")
                    break
                
                # Filter out already processed URLs
                new_items = []
                for item in batch_items:
                    url = item.get("link", "")
                    if url and url not in processed_urls:
                        new_items.append(item)
                
                print(f"Got {len(batch_items)} results, {len(new_items)} are new URLs")
                all_items.extend(new_items)
                
                # If we have enough unique URLs, we can stop fetching
                if len(all_items) >= total_needed:
                    print(f"Collected enough unique URLs: {len(all_items)}")
                    break
                
                start_index += 10
                
            except Exception as e:
                print(f"Error fetching batch {batch_num + 1}: {str(e)}")
                break
        
        return all_items

    # Main processing logic with retry mechanism
    successful_results = []
    processed_urls = set()
    total_attempts = 0
    
    print(f"Starting search for '{query}' - targeting {num_results} successful results")
    
    while len(successful_results) < num_results and total_attempts < max_retry_attempts:
        # Calculate how many more results we need
        remaining_needed = num_results - len(successful_results)
        
        # Fetch multiple batches of search results to get enough unique URLs
        search_items = await fetch_multiple_search_batches(
            query=query,
            total_needed=remaining_needed * 2,  # Fetch 2x more to account for failures
            processed_urls=processed_urls,
            language=language,
            country=country,
            safe_search=safe_search
        )
        
        if not search_items:
            print("No more search results available")
            break
            
        print(f"Retrieved {len(search_items)} unique search results to process")
        
        # Process results
        if concurrent_scraping and use_web_scraper:
            # Process results concurrently with controlled concurrency
            semaphore = asyncio.Semaphore(max_concurrent_scrapers)
            
            async def process_with_semaphore(item):
                async with semaphore:
                    return await process_single_result(item, processed_urls)
            
            # Create tasks for concurrent processing
            tasks = [process_with_semaphore(item) for item in search_items]
            
            # Process tasks and collect successful results
            for coro in asyncio.as_completed(tasks):
                if len(successful_results) >= num_results:
                    break
                    
                try:
                    result = await coro
                    if result is not None:
                        successful_results.append(result)
                        print(f"Progress: {len(successful_results)}/{num_results} successful extractions")
                except Exception as e:
                    print(f"Task failed: {str(e)}")
                    
        else:
            # Sequential processing
            for item in search_items:
                if len(successful_results) >= num_results:
                    break
                    
                result = await process_single_result(item, processed_urls)
                if result is not None:
                    successful_results.append(result)
                    print(f"Progress: {len(successful_results)}/{num_results} successful extractions")
        
        total_attempts = len(processed_urls)
        print(f"Batch complete. Successful: {len(successful_results)}, Total attempted: {total_attempts}")
        
        # If we still need more results and haven't hit the retry limit, continue
        if len(successful_results) < num_results and total_attempts < max_retry_attempts:
            print(f"Still need {num_results - len(successful_results)} more results, continuing search...")
    
    print(f"Search complete. Successfully extracted {len(successful_results)} out of {num_results} requested results")
    
    # Return the full result dictionaries containing title, link, and extracted_content
    return successful_results


# Create the enhanced Google search tool with web scraper integration
google_search_with_scraper_tool = FunctionTool(
    func=google_search_with_scraper,
    description="""
    Perform Google searches using the Custom Search API with intelligent web content extraction.
    Uses SmartScraperGraph to extract query-relevant information from each webpage.
    Automatically retries with additional search results if extractions fail.
    Supports searching unlimited results by making multiple paginated API calls.
    Requires GOOGLE_API_KEY, GOOGLE_CSE_ID, and GEMINI_API_KEY environment variables.
    """,
    global_imports=[
        ImportFromModule("typing", ("List", "Dict", "Optional")),
        "os",
        "httpx",
        "html2text",
        ImportFromModule("bs4", ("BeautifulSoup",)),
        ImportFromModule("urllib.parse", ("urljoin",)),
        ImportFromModule("scrapegraphai.graphs", ("SmartScraperGraph",)),
    ],
)

if __name__ == "__main__":
    import asyncio
    import json

    async def main():
        queries = [
            "What are the latest gene editing techniques used in cancer research?",
            # "How do BRCA1 and BRCA2 mutations influence breast cancer risk?",
            # "What is the role of TP53 gene in tumor suppression?",
            # "How are oncogenes and tumor suppressor genes different?",
            # "What are the current gene therapy approaches for treating leukemia?",
            # "How does gene expression profiling help in cancer diagnosis?",
            # "What is the impact of epigenetic changes on cancer progression?",
            # "How are targeted therapies developed for specific cancer gene mutations?",
            # "What are the challenges in using CRISPR for cancer treatment?",
            # "How do genetic biomarkers guide personalized cancer therapies?",
        ]
        all_results = []
        for query in queries:
            print(f"Running Google search for: {query}")
            results = await google_search_with_scraper(
                query=query,
                num_results=20,  # Now supports more than 10 results!
                include_snippets=True,
                use_web_scraper=True,
                concurrent_scraping=True,
                max_concurrent_scrapers=10,
                max_retry_attempts=40,  # Increased to accommodate larger searches
                language="en",
                country="us",
                safe_search=True,
            )
            
            # Save results as JSON
            all_results.extend(results)

        
        with open("google_search_scraper_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to google_search_scraper_results.json - {len(results)} successful extractions")

    asyncio.run(main())
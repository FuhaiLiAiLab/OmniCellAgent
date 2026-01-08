"""
medical_research_rag.py - A module for querying medical research papers using RAG
"""
import re
import os
import sys
import json
import asyncio
from paperscraper.pubmed import get_and_dump_pubmed_papers
from paperscraper.pdf import save_pdf_from_dump

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Handle both relative and absolute imports for prompt
try:
    from utils.prompt import PUBMED_AGENT_SYSTEM_MESSAGE_v1
    from utils.path_config import get_path
except ImportError:
    # Try relative import from current location
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.prompt import PUBMED_AGENT_SYSTEM_MESSAGE_v1
    from utils.path_config import get_path

# Handle both relative and absolute imports for tools
try:
    from .get_papers_info_tools import get_papers_info, is_error_file
except ImportError:
    from get_papers_info_tools import get_papers_info, is_error_file
from typing import List, Dict, Any
import time

from dotenv import load_dotenv

# AutoGen imports for agent functionality
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import StructuredMessage
    from autogen_core.models import ModelInfo
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("AutoGen not available. Install with: pip install autogen-agentchat autogen-ext[openai]")

load_dotenv('../.env')  

# Constants - use path config
DEFAULT_JSONL_CACHE_DIR = get_path('cache.pubmed_jsonl', absolute=True, create=True)
DEFAULT_DOI_CACHE_DIR = get_path('cache.pubmed_doi', absolute=True, create=True)
DEFAULT_TOP_K = 3


class MedicalResearchProcessor: # Renamed class as it no longer handles RAG
    """Class to handle medical research paper retrieval and processing"""

    def __init__(self,
                 jsonl_cache_dir: str = DEFAULT_JSONL_CACHE_DIR,
                 doi_cache_dir: str = DEFAULT_DOI_CACHE_DIR):
        """
        Initialize the MedicalResearchProcessor.

        Args:
            jsonl_cache_dir: Directory for caching paper metadata
            doi_cache_dir: Directory for caching PDF files
        """
        self.jsonl_cache_dir = jsonl_cache_dir
        self.doi_cache_dir = doi_cache_dir
       
        for directory in [jsonl_cache_dir, doi_cache_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)



    def cache_paper(self, query: List[str], top_k: int = DEFAULT_TOP_K, timestr: str = None) -> str:
        """
        Cache papers from PubMed using incremental downloading to reach the target number.

        Args:
            query: List of query terms
            top_k: Number of papers to retrieve (will incrementally download until target is reached)

        Returns:
            Path to the directory containing cached papers
        """
        pdf_dir = os.path.join(self.doi_cache_dir, timestr)
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Create temp directory for single-paper jsonl files
        temp_dir = os.path.join(self.jsonl_cache_dir, f"{timestr}_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        readable_papers = 0
        attempt = 1
        max_attempts = 3
        fetch_multiplier = 100
        all_papers_metadata = []  # Store all fetched metadata
        
        while readable_papers < top_k and attempt <= max_attempts:
            print(f"[Attempt {attempt}] Fetching papers to reach {top_k} readable papers...")
            
            # Calculate how many more papers we need
            papers_needed = top_k - readable_papers
            fetch_count = max(papers_needed, int(papers_needed * fetch_multiplier))
            
            # Get and dump papers for this attempt
            output_filepath = os.path.join(self.jsonl_cache_dir, f"{timestr}_attempt_{attempt}.jsonl")
            
            try:
                get_and_dump_pubmed_papers(
                    query, 
                    output_filepath=output_filepath, 
                    max_results=fetch_count
                )
                
                # Load new papers from this attempt
                with open(output_filepath, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        print(f"[!] No papers found for query: {query} on attempt {attempt}")
                        break
                
                # Parse and add new papers to our collection
                new_papers = []
                for line in lines:
                    try:
                        paper_data = json.loads(line.strip())
                        if paper_data.get('doi'):  # Only include papers with DOI
                            new_papers.append(paper_data)
                    except:
                        continue
                
                all_papers_metadata.extend(new_papers)
                print(f"[Attempt {attempt}] Found {len(new_papers)} papers with DOI in metadata")
                
                # Use incremental downloading for new papers
                downloaded_count = self._incremental_download(
                    new_papers, pdf_dir, temp_dir, timestr, readable_papers, top_k
                )
                
                # Update readable papers count
                readable_papers = self._count_readable_files(pdf_dir)
                
                print(f"[Attempt {attempt}] Successfully downloaded {downloaded_count} new papers")
                print(f"[Total] {readable_papers}/{top_k} readable papers available")
                
                if readable_papers >= top_k:
                    print(f"Successfully reached target of {top_k} readable papers!")
                    break
                else:
                    print(f"Still need {top_k - readable_papers} more readable papers...")
                    fetch_multiplier += 0.5
                
            except FileNotFoundError:
                print(f"[!] Metadata file not found: {output_filepath}")
                break
            except Exception as e:
                print(f"[!] Error in attempt {attempt}: {e}")
            
            attempt += 1
        
        # Create consolidated metadata file with all collected papers FIRST
        consolidated_success = self._create_consolidated_metadata(all_papers_metadata, timestr)
        
        # Clean up temporary files (always safe to cleanup)
        self._cleanup_temp_files(temp_dir)
        
        # Only cleanup attempt files if consolidated file was created successfully
        if consolidated_success:
            self._cleanup_attempt_files(timestr, attempt - 1)
        else:
            print(f"[Warning] Keeping attempt files as backup since consolidated file creation failed")
        
        # Final summary
        final_count = self._count_readable_files(pdf_dir)
        if final_count < top_k:
            print(f"Warning: Only found {final_count}/{top_k} readable papers after {attempt-1} attempts")
            print(f"   Some papers may be behind paywalls, inaccessible, or contain errors")
        
        return pdf_dir

    def _incremental_download(self, papers_metadata: List[Dict], pdf_dir: str, temp_dir: str, 
                             timestr: str, current_readable: int, target_count: int) -> int:
        """
        Download papers incrementally using individual temp .jsonl files.
        
        Args:
            papers_metadata: List of paper metadata dictionaries
            pdf_dir: Directory to save PDFs
            temp_dir: Temporary directory for single-paper jsonl files
            timestr: Timestamp string for naming
            current_readable: Current count of readable papers
            target_count: Target number of papers needed
            
        Returns:
            Number of successfully downloaded papers
        """
        downloaded_count = 0
        
        for i, paper_data in enumerate(papers_metadata):
            # Stop if we've reached our target
            if current_readable + downloaded_count >= target_count:
                print(f"[Info] Reached target of {target_count} papers, stopping download")
                break
                
            doi = paper_data.get('doi', '')
            if not doi:
                continue
                
            # Check if paper already exists (PDF or XML)
            filename = doi.replace("/", "_")
            pdf_file = os.path.join(pdf_dir, f"{filename}.pdf")
            xml_file = os.path.join(pdf_dir, f"{filename}.xml")
            
            if os.path.exists(pdf_file) or os.path.exists(xml_file):
                print(f"[Skip] Paper {i+1}: {filename} already exists")
                continue
            
            # Create temporary single-paper jsonl file
            temp_jsonl_path = os.path.join(temp_dir, f"{timestr}_paper_{i+1}_{filename[:50]}.jsonl")
            
            try:
                with open(temp_jsonl_path, 'w') as temp_file:
                    json.dump(paper_data, temp_file)
                    temp_file.write('\n')
                
                print(f"[Download] Paper {i+1}/{len(papers_metadata)}: Attempting {doi}")
                
                # Track download success before attempting
                before_files = set(os.listdir(pdf_dir)) if os.path.exists(pdf_dir) else set()
                
                # Download using the single-paper temp file
                save_pdf_from_dump(temp_jsonl_path, pdf_path=pdf_dir, key_to_save='doi')
                
                # Check if download was successful
                after_files = set(os.listdir(pdf_dir)) if os.path.exists(pdf_dir) else set()
                new_files = after_files - before_files
                
                if new_files:
                    # Verify the downloaded file is readable
                    success = False
                    for new_file in new_files:
                        file_path = os.path.join(pdf_dir, new_file)
                        if new_file.endswith('.pdf'):
                            try:
                                if os.path.getsize(file_path) > 1024:  # At least 1KB
                                    success = True
                                    break
                            except:
                                continue
                        elif new_file.endswith('.xml'):
                            # Check if it's not an error file
                            try:
                                if not is_error_file(file_path):
                                    success = True
                                    break
                            except:
                                continue
                    
                    if success:
                        downloaded_count += 1
                        print(f"[Success] Paper {i+1}: Downloaded successfully ({downloaded_count} total)")
                    else:
                        print(f"[Failed] Paper {i+1}: Downloaded but file appears corrupted")
                else:
                    print(f"[Failed] Paper {i+1}: No file downloaded")
                
            except Exception as e:
                print(f"[Error] Paper {i+1}: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_jsonl_path):
                    try:
                        os.remove(temp_jsonl_path)
                    except:
                        pass
        
        return downloaded_count

    def _cleanup_temp_files(self, temp_dir: str):
        """Clean up temporary directory and any remaining temp files."""
        if os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"[Cleanup] Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"[Warning] Could not clean up temp directory {temp_dir}: {e}")

    def _cleanup_attempt_files(self, timestr: str, num_attempts: int):
        """Clean up attempt-specific metadata files."""
        for attempt in range(1, num_attempts + 1):
            attempt_filepath = os.path.join(self.jsonl_cache_dir, f"{timestr}_attempt_{attempt}.jsonl")
            if os.path.exists(attempt_filepath):
                try:
                    os.remove(attempt_filepath)
                    print(f"[Cleanup] Removed attempt file: {attempt_filepath}")
                except Exception as e:
                    print(f"[Warning] Could not clean up attempt file {attempt_filepath}: {e}")

    def _create_consolidated_metadata(self, all_papers_metadata: List[Dict], timestr: str) -> bool:
        """
        Create a consolidated metadata file from all collected papers.
        
        Returns:
            bool: True if successful, False otherwise
        """
        consolidated_filepath = os.path.join(self.jsonl_cache_dir, f"{timestr}.jsonl")
        
        try:
            with open(consolidated_filepath, 'w') as consolidated_file:
                seen_dois = set()  # Avoid duplicates
                
                for paper_data in all_papers_metadata:
                    doi = paper_data.get('doi', '')
                    if doi and doi not in seen_dois:
                        json.dump(paper_data, consolidated_file)
                        consolidated_file.write('\n')
                        seen_dois.add(doi)
            
            print(f"[Info] Consolidated metadata saved to: {consolidated_filepath}")
            print(f"[Info] Total unique papers in metadata: {len(seen_dois)}")
            return True
            
        except Exception as e:
            print(f"[Error] Could not create consolidated metadata file: {e}")
            return False

    def _count_readable_files(self, pdf_dir: str) -> int:
        """Count successfully downloaded AND readable PDF and XML files"""
        if not os.path.exists(pdf_dir):
            return 0
        
        files = os.listdir(pdf_dir)
        readable_count = 0
        
        for file in files:
            if file.endswith('.pdf'):
                # For PDF files, just check if they exist and have content
                file_path = os.path.join(pdf_dir, file)
                try:
                    if os.path.getsize(file_path) > 1024:  # At least 1KB for a valid PDF
                        readable_count += 1
                    else:
                        print(f"[Debug] Skipping small/empty PDF file: {file}")
                except Exception as e:
                    print(f"[Debug] Error checking PDF file {file}: {e}")
                    
            elif file.endswith('.xml'):
                # For XML files, check for error content
                file_path = os.path.join(pdf_dir, file)
                if not is_error_file(file_path):
                    readable_count += 1
                else:
                    print(f"[Debug] Skipping error XML file: {file}")
        
        return readable_count
    
    async def load_papers(self, paper_dir: str, use_llm_processing: bool = True, max_concurrent: int = 10, target_count: int = None) -> List[Dict[str, Any]]:
        """
        Load papers from a directory (handles PDF and XML) with optional LLM processing.

        Args:
            paper_dir: Directory containing papers
            use_llm_processing: Whether to use LLM for enhanced content processing
            max_concurrent: Maximum concurrent LLM operations
            target_count: Maximum number of papers to process (None for all)

        Returns:
            List of paper dictionaries with processed content
        """
        # Extract timestamp from paper_dir to find corresponding JSONL file
        jsonl_file_path = None
        if paper_dir:
            # Extract the timestamp from the paper directory path
            dir_name = os.path.basename(paper_dir)
            
            # Look for consolidated JSONL file with the same timestamp
            consolidated_jsonl = os.path.join(self.jsonl_cache_dir, f"{dir_name}.jsonl")
            
            if os.path.exists(consolidated_jsonl):
                jsonl_file_path = consolidated_jsonl
                print(f"[Info] Found consolidated JSONL metadata: {consolidated_jsonl}")
            else:
                # If consolidated file doesn't exist, look for attempt files
                print(f"[Warning] Consolidated JSONL metadata file not found: {consolidated_jsonl}")
                
                # Try to find any attempt files with the same timestamp
                attempt_files = []
                if os.path.exists(self.jsonl_cache_dir):
                    for file in os.listdir(self.jsonl_cache_dir):
                        if file.startswith(f"{dir_name}_attempt_") and file.endswith('.jsonl'):
                            attempt_files.append(os.path.join(self.jsonl_cache_dir, file))
                
                if attempt_files:
                    # Use the first (or most recent) attempt file
                    jsonl_file_path = attempt_files[0]
                    print(f"[Info] Using attempt file as fallback: {jsonl_file_path}")
                else:
                    # Last resort: look for any JSONL file with similar timestamp
                    print(f"[Warning] No metadata files found for timestamp {dir_name}")
                    if os.path.exists(self.jsonl_cache_dir):
                        all_jsonl_files = [f for f in os.listdir(self.jsonl_cache_dir) if f.endswith('.jsonl')]
                        if all_jsonl_files:
                            # Sort by modification time and use the most recent
                            all_jsonl_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.jsonl_cache_dir, x)), reverse=True)
                            jsonl_file_path = os.path.join(self.jsonl_cache_dir, all_jsonl_files[0])
                            print(f"[Info] Using most recent JSONL file as fallback: {jsonl_file_path}")
        
        papers = await get_papers_info(
            paper_dir,
            jsonl_file_path=jsonl_file_path,
            use_llm_processing=use_llm_processing,
            max_concurrent=max_concurrent
        )
        
        # Limit to target count if specified
        if target_count is not None and len(papers) > target_count:
            print(f"[Info] Limiting results to {target_count} papers out of {len(papers)} processed")
            papers = papers[:target_count]
        
        return papers

async def query_medical_research_async(
    query: str,
    top_k: int = 3,
    use_llm_processing: bool = True,
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve and process medical research papers.

    Args:
        query(str): The medical topic to search for (e.g., "COVID-19 Treatment")
        top_k(int): Number of readable papers to retrieve (default: 3)
        use_llm_processing(bool): Whether to use LLM for enhanced content processing (default: True)
        max_concurrent(int): Maximum concurrent LLM operations (default: 10)

    Returns:
        List[Dict]: Information about retrieved papers and their content.
    """
    try:
        timestr = time.strftime("%Y%m%d-%H%M%S") 
        processor = MedicalResearchProcessor() # Use the refactored class

        # Cache papers (now ensures readable papers)
        print(f"Caching top {top_k} readable papers for query: '{query}'")
        paper_dir = processor.cache_paper([query], top_k=top_k,
                                          timestr=timestr)

        if not paper_dir:
             return f"Failed to cache papers for query: {query}. Check logs for details."

        # Load papers with optional LLM processing
        print(f"Loading papers from: {paper_dir}")
        print(f"LLM processing: {'Enabled' if use_llm_processing else 'Disabled'}")
        if use_llm_processing:
            print(f"Max concurrent operations: {max_concurrent}")
            
        paper_info = await processor.load_papers(
            paper_dir, 
            use_llm_processing=use_llm_processing,
            max_concurrent=max_concurrent,
            target_count=top_k
        )
        return paper_info

    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        return f"An error occurred while processing medical research: {str(e)}"


async def query_medical_research(
    query: str,
    top_k: int = 3,
    use_llm_processing: bool = True,
    max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for retrieving and processing medical research papers.

    Args:
        query(str): The medical topic to search for (e.g., "COVID-19 Treatment")
        top_k(int): Number of readable papers to retrieve (default: 3)
        use_llm_processing(bool): Whether to use LLM for enhanced content processing (default: True)
        max_concurrent(int): Maximum concurrent LLM operations (default: 10)

    Returns:
        List[Dict]: Information about retrieved papers and their content.
    """

    return await query_medical_research_async(query, top_k, use_llm_processing, max_concurrent)


# AutoGen Agent Functions
async def medical_research_tool(
    query: str,
    top_k: int = 10,
    use_llm_processing: bool = True,
    max_concurrent: int = 10
) -> str:
    """
    Search and retrieve medical research papers from PubMed as an AutoGen tool.
    
    Args:
        query: The medical topic to search for (e.g., "COVID-19 Treatment", "leptin resistance obesity")
        top_k: Number of readable papers to retrieve (default: 10, recommended range: 1-20)
        use_llm_processing: Whether to use LLM for enhanced content processing (default: True)
        max_concurrent: Maximum concurrent LLM operations (default: 10)
    
    Returns:
        Formatted string containing information about retrieved papers and their content
    """

    top_k=30

    try:
        # Call the existing async function
        papers = await query_medical_research_async(
            query=query,
            top_k=top_k,
            use_llm_processing=use_llm_processing,
            max_concurrent=max_concurrent
        )
        
        if isinstance(papers, str):
            # Error case
            return f"Error retrieving papers: {papers}"
        
        if not papers:
            return f"No papers found for query: '{query}'"
        
        # Format the results for the agent
        result_lines = [
            f"Found {len(papers)} medical research papers for query: '{query}'",
            "=" * 60
        ]
        
        for i, paper in enumerate(papers, 1):
            paper_info = [
                f"\nPaper {i}:",
                f"  Title: {paper.get('title', 'N/A')}",
                f"  Has LLM Processing: {'Yes' if paper.get('llm_content') else 'No'}"
            ]
            
            # Add LLM-processed content summary if available
            if paper.get('llm_content'):
                llm_summary = paper['llm_content']
                paper_info.append(f"  Content: {llm_summary}")
            else:
                paper_info.append(f"  Content: Title only (no content processing)")
            
            result_lines.extend(paper_info)
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"Error in medical research tool: {str(e)}"

async def run_medical_research_agent(
    task: str,
) -> None:
    """
    Run the medical research agent with a given task.
    
    Args:
        task: The medical research task/question
        model_name: OpenAI model to use
        api_key: OpenAI API key
        output_stats: Whether to show execution statistics
    """
    if not AUTOGEN_AVAILABLE:
        raise ImportError("AutoGen is not available. Install with: pip install autogen-agentchat autogen-ext[openai]")
    
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-pro",
        model_info=ModelInfo(
            vision=True, 
            function_calling=True, 
            json_output=True, 
            family="GEMINI_2_5_PRO", 
            structured_output=False
        ),
    )

    system_message = PUBMED_AGENT_SYSTEM_MESSAGE_v1

    # Create and return the agent
    agent = AssistantAgent(
        name="medical_research_assistant",
        model_client=model_client,
        tools=[medical_research_tool],
        system_message=system_message,
        reflect_on_tool_use=True,
    )
    
    # Run the agent with console output
    await Console(
        agent.run_stream(task=task),
    )


# Example usage
if __name__ == "__main__":

    async def main():
        # Example 1: Basic usage with LLM processing
        result = await query_medical_research_async(
            query="What are the key dysfunctional signaling targets in microglia of AD?",
            top_k=10,
            use_llm_processing=True,
            max_concurrent=10
        )
        
    async def autogen_example():        
        await run_medical_research_agent(
            task="What are the key dysfunctional signaling targets in microglia of AD?"
        )
        
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "autogen":
        try:
            asyncio.run(autogen_example())
        except RuntimeError:
            print("An error occurred: The event loop is already running. Please ensure this script is run in a proper async context.")
    else:
        try:
            asyncio.run(main())
        except RuntimeError:
            print("An error occurred: The event loop is already running. Please ensure this script is run in a proper async context.")
            print("\nTo run AutoGen examples, use: python query_pubmed_tool.py autogen")
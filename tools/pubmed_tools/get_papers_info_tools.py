from typing import List, Dict, Any 
import os
import re
import pymupdf
import base64
import asyncio # Added import
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import time
import json
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .get_elsiver_paper import *
    from .get_pmc_paper import *
    from .get_paper_summary import get_paper_summary, has_error_content, get_paper_summary_async
except ImportError:
    from get_elsiver_paper import *
    from get_pmc_paper import *
    from get_paper_summary import get_paper_summary, has_error_content, get_paper_summary_async


ELSIVER_ID = "http://www.elsevier.com/xml/"
PMC_ID = """key="article-id_pmc"""

def is_error_file(file_path: str) -> bool:
    """
    Check if the file contains error messages indicating failed download/parsing.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return has_error_content(content)
    except Exception as e:
        print(f"[Error] Could not check file {file_path}: {e}")
        return True  # Treat unreadable files as errors

def extract_and_validate_json(content: str) -> tuple[dict, bool]:
    """
    Extract and validate JSON content from LLM output.
    Since get_paper_summary.py now handles markdown extraction and syntax fixing,
    this should receive clean JSON.
    
    Returns:
        tuple: (parsed_json_or_original_content, is_valid_json)
    """
    if not content:
        return content, False
    
    # Try direct JSON parsing
    try:
        return json.loads(content), True
    except json.JSONDecodeError as e:
        print(f"[Warning] Could not parse JSON from LLM response: {e}")
        # Return original content as fallback
        return content, False

def load_paper_metadata_from_jsonl(jsonl_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load paper metadata from JSONL file and create a mapping based on DOI.
    
    Args:
        jsonl_file_path: Path to the JSONL file containing paper metadata
        
    Returns:
        Dictionary mapping DOI to paper metadata
    """
    metadata_map = {}
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        paper_data = json.loads(line)
                        doi = paper_data.get('doi', '')
                        if doi:
                            # Convert DOI to filename format (replace / with _)
                            filename_doi = doi.replace("/", "_")
                            metadata_map[filename_doi] = paper_data
                    except json.JSONDecodeError as e:
                        print(f"[Warning] Failed to parse JSON line: {e}")
                        continue
                        
    except FileNotFoundError:
        print(f"[Warning] JSONL file not found: {jsonl_file_path}")
    except Exception as e:
        print(f"[Error] Failed to load JSONL metadata: {e}")
    
    print(f"[Info] Loaded metadata for {len(metadata_map)} papers from JSONL")
    return metadata_map

async def process_single_paper(file_path: str, file_name: str, paper_metadata: Dict[str, Any], use_llm_processing: bool = False, semaphore=None) -> Dict[str, Any]:
    """
    Process a single paper file asynchronously.
    """
    if semaphore:
        async with semaphore:  # Limit concurrent LLM calls
            return await _process_single_paper_internal(file_path, file_name, paper_metadata, use_llm_processing)
    else:
        return await _process_single_paper_internal(file_path, file_name, paper_metadata, use_llm_processing)

async def _process_single_paper_internal(file_path: str, file_name: str, paper_metadata: Dict[str, Any], use_llm_processing: bool) -> Dict[str, Any]:
    """
    Internal method to process a single paper.
    """
    paper_entry = {
        "title": paper_metadata.get("title", "Unknown Title"),
        "authors": paper_metadata.get("authors", []),
        "date": paper_metadata.get("date", "Unknown Date"),
        "llm_content": None
    }
    
    try:
        if file_name.endswith('.xml'):
            print(f"[Debug] Processing XML file: {file_name}")
            
            # Read file asynchronously
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                xml_content = await f.read()
            
            # First check if the content contains error messages
            if has_error_content(xml_content):
                return None  # Return None for failed papers
            
            if ELSIVER_ID in xml_content:
                print(f"XML file '{file_path}' contains Elsevier content.")
                sections = get_article_sections_from_xml(xml_content)
                if sections:
                    text = format_sections_for_display(sections)
                    if not has_error_content(text) and len(text.strip()) > 50:
                        if use_llm_processing:
                            processed_text = await get_paper_summary_async(text=text, mine_type="text/plain")
                            if processed_text:
                                paper_entry['llm_content'] = processed_text
                    else:
                        return None  # Return None for failed papers
                else:
                    return None  # Return None for failed papers

            elif PMC_ID in xml_content:
                print(f"XML file '{file_path}' contains PMC content.")
                text = parse_xml_to_sections(file_path)
                if not has_error_content(text) and len(text.strip()) > 50:
                    if use_llm_processing:
                        processed_text = await get_paper_summary_async(text=text, mine_type="text/plain")
                        if processed_text:
                            paper_entry['llm_content'] = processed_text
                else:
                    return None  # Return None for failed papers
            else:
                return None  # Return None for failed papers

        elif file_name.endswith('.pdf'):
            print(f"[Debug] Processing PDF file: {file_path}")
            
            # Always extract raw text first
            loop = asyncio.get_event_loop()
            extracted_text = await loop.run_in_executor(
                None, 
                lambda: chr(12).join([page.get_text() for page in pymupdf.open(file_path)])
            )
            
            if not has_error_content(extracted_text) and len(extracted_text.strip()) > 50:
                if use_llm_processing:
                    try:
                        processed_text = await get_paper_summary_async(file_path=file_path, mine_type="application/pdf")
                        if processed_text and not has_error_content(processed_text) and len(processed_text.strip()) > 50:
                            paper_entry['llm_content'] = processed_text
                            print(f"[Debug] Successfully processed pdf with LLM: {file_name}")
                    except Exception as e:
                        print(f"[Debug] LLM processing failed for {file_name}: {e}, using raw content only")
                
                print(f"[Debug] Successfully processed pdf: {file_name}")
            else:
                return None  # Return None for failed papers
        else:
            return None  # Return None for failed papers
            
    except Exception as e:
        print(f"[Error] Failed to process {file_name}: {e}")
        return None  # Return None for failed papers
    
    return paper_entry

async def get_papers_info(paper_dir: str, jsonl_file_path: str = None, use_llm_processing: bool = False, max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Process papers from a directory in parallel and extract their content.
    
    Args:
        paper_dir: Directory containing paper files (PDF/XML)
        jsonl_file_path: Path to JSONL file containing paper metadata (optional)
        use_llm_processing: Whether to use LLM for content processing and formatting
        max_concurrent: Maximum number of concurrent LLM calls to prevent rate limiting
    
    Returns:
        List of paper dictionaries with valid content
    """
    start_time = time.time()
    print(f"[Info] Starting parallel processing of papers in {paper_dir}")
    print(f"[Info] LLM processing: {'Enabled' if use_llm_processing else 'Disabled'}")
    print(f"[Info] Max concurrent operations: {max_concurrent}")
    
    # Load metadata from JSONL if provided
    metadata_map = {}
    if jsonl_file_path and os.path.exists(jsonl_file_path):
        metadata_map = load_paper_metadata_from_jsonl(jsonl_file_path)
    else:
        print(f"[Warning] No JSONL metadata file provided or file doesn't exist: {jsonl_file_path}")
    
    # Get all valid paper files
    all_files = os.listdir(paper_dir)
    paper_files = []
    
    for file_name in all_files:
        if file_name.endswith(('.xml', '.pdf')):
            file_path = os.path.join(paper_dir, file_name)
            
            # Extract DOI from filename (remove extension and convert back)
            filename_base = os.path.splitext(file_name)[0]
            
            # Find matching metadata
            paper_metadata = metadata_map.get(filename_base, {"title": f"Unknown Title ({file_name})"})
            
            paper_files.append((file_name, file_path, paper_metadata))
    
    print(f"[Info] Found {len(paper_files)} paper files to process")
    print(f"[Info] {len([f for f in paper_files if f[2].get('title', '').startswith('Unknown')])} files without metadata")
    
    # Create semaphore to limit concurrent LLM calls
    semaphore = asyncio.Semaphore(max_concurrent) if use_llm_processing else None
    
    # Create tasks for parallel processing
    tasks = []
    for file_name, file_path, paper_metadata in paper_files:
        task = process_single_paper(file_path, file_name, paper_metadata, use_llm_processing, semaphore)
        tasks.append(task)
    
    # Process all papers in parallel
    print(f"[Info] Processing {len(tasks)} papers in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter and collect successful results
    paper_info_list = []
    successful_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[Error] Task {i} failed with exception: {result}")
            error_count += 1
            continue
            
        if result is not None:  # Successfully processed paper
            paper_info_list.append(result)
            successful_count += 1
            print(f"[Debug] Successfully added paper {successful_count}: {paper_files[i][0]}")
        else:
            skipped_count += 1
            print(f"[Debug] Skipped {paper_files[i][0]}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"[Summary] Processing completed in {processing_time:.2f} seconds")
    print(f"[Summary] Successfully processed: {successful_count} papers")
    print(f"[Summary] Skipped: {skipped_count} papers") 
    print(f"[Summary] Errors: {error_count} papers")
    
    
    # Save results to JSON file
    if paper_info_list:
        await save_results_to_json(paper_info_list, paper_dir, processing_time, successful_count, skipped_count, error_count)
    
    return paper_info_list

async def save_results_to_json(paper_info_list: List[Dict[str, Any]], paper_dir: str, processing_time: float, successful_count: int, skipped_count: int, error_count: int):
    """
    Save the processed papers to a JSON file with metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/logs/processed_papers_{timestamp}.json"
    
    # Prepare the output structure
    output_data = []
    
    # Process each paper for JSON output
    for paper in paper_info_list:
        # Process LLM content to ensure it's well-structured
        llm_content = paper.get("llm_content", None)
        processed_content = None
        
        if llm_content:
            # Use the robust JSON extraction function
            processed_content, is_valid_json = extract_and_validate_json(llm_content)
            
            if not is_valid_json:
                print(f"[Warning] Could not parse JSON for paper '{paper['title']}', keeping as text")
            else:
                print(f"[Debug] Successfully parsed JSON for paper '{paper['title']}'")
        else:
            processed_content = None
        
        paper_data = {
            "title": paper["title"],
            "authors": paper.get("authors", []),
            "date": paper.get("date", "Unknown Date"),
            "content": processed_content,
        }
        output_data.append(paper_data)
    
    # Save to JSON file
    try:
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(output_data, indent=2, ensure_ascii=False))
        
        print(f"\n[JSON Export] Results saved to: {output_file}")
        print(f"[JSON Export] Total papers in file: {len(paper_info_list)}")
        print(f"[JSON Export] Papers with structured LLM content: {sum(1 for p in output_data if p.get('content') and isinstance(p['content'], dict))}")
        print(f"[JSON Export] Papers with text LLM content: {sum(1 for p in output_data if p.get('content') and isinstance(p['content'], str))}")
        print(f"[JSON Export] Papers with titles only: {sum(1 for p in output_data if not p.get('content'))}")
        
    except Exception as e:
        print(f"[Error] Failed to save JSON file: {e}")
        # Fallback to sync write
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"[JSON Export] Results saved to: {output_file} (fallback mode)")


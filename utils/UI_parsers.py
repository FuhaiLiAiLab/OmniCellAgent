import gradio as gr
import json
import re
import html
from datetime import datetime
from typing import Any, Dict, List, Union
import ast
import shutil



def extract_function_result_content(text: str) -> str:
    """Extract content from FunctionExecutionResult wrapper"""
    content_match = re.search(r"content='(.*?)', name=", text, re.DOTALL)
    if content_match:
        return content_match.group(1)
    
    content_match = re.search(r"content=(.*?), name=", text, re.DOTALL)
    if content_match:
        return content_match.group(1)
    
    return text

def safe_parse_content(content_str: str) -> Any:
    """Safely parse string content, handling various formats"""
    try:
        return ast.literal_eval(content_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(content_str)
        except json.JSONDecodeError:
            return content_str

def convert_search_results(input_text: str) -> Dict[str, Any]:
    """Convert the search results format to clean JSON"""
    content = extract_function_result_content(input_text)
    parsed_content = safe_parse_content(content)
    if parsed_content.startswith("##"):
        return parsed_content
    
    if parsed_content.endswith(".csv')\", name='fetch_data_from_database', call_id='', is_error=False)]"):
        return parsed_content
    
    if parsed_content.startswith("<table"):
        return parsed_content
    if isinstance(parsed_content, list):
        return {
            "search_results": parsed_content,
            "metadata": {
                "total_results": len(parsed_content),
                "processed_at": "auto-generated",
                "format": "converted_from_function_result"
            }
        }
    else:
        return {
            "raw_content": parsed_content,
            "metadata": {
                "format": "unparsable_content",
                "processed_at": "auto-generated"
            }
        }


def clean_and_parse_data(data: Union[str, Dict, list]) -> Union[Dict, list]:
    """
    Clean heavily escaped JSON strings and convert to Python objects
    """
    if isinstance(data, (dict, list)):
        return data
    
    if isinstance(data, str):
        # Multiple rounds of HTML unescaping may be needed
        cleaned = data
        for _ in range(5):  # Up to 5 rounds of unescaping
            old_cleaned = cleaned
            cleaned = html.unescape(cleaned)
            if cleaned == old_cleaned:
                break  # No more changes
        
        # Remove surrounding quotes if present
        cleaned = cleaned.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        
        try:
            # Try to parse as JSON
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # If that fails, return the cleaned string
            return cleaned
    
    return data


def generate_search_results_content(data: Union[str, dict, list]) -> str:
    """
    Generate HTML content using JSON pretty printing for clean formatting
    """
    try:
        # If it's a string, try to parse it as JSON
        if isinstance(data, str):


            # Clean up escaped quotes first
            cleaned_data = data.strip()
            if cleaned_data.startswith('"') and cleaned_data.endswith('"'):
                cleaned_data = cleaned_data[1:-1]  # Remove outer quotes
            if data.startswith("<table"):
                return data
            if data.startswith("##"):
                return data # markdown
            # Parse the JSON
            parsed_data = json.loads(cleaned_data)
        else:
            parsed_data = data
        
        # Pretty print the JSON with nice indentation
        json_string = json.dumps(parsed_data, indent=2, ensure_ascii=False, default=str)
        
        # Return as formatted HTML
        return f"""
        <div style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #f8f9fa;">
            <div style="font-weight: bold; color: #333; margin-bottom: 10px; font-size: 1.1em;">
                📄 Search Results Data
            </div>
            <pre style="background: #fff; padding: 15px; border-radius: 4px; border: 1px solid #e0e0e0; 
                       overflow-x: auto; font-family: 'Courier New', monospace; font-size: 0.9em; 
                       line-height: 1.4; color: #333; margin: 0; white-space: pre-wrap; 
                       word-wrap: break-word;">{html.escape(json_string)}</pre>
        </div>
        """
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, display as plain text
        return f"""
        <div style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #fff3cd;">
            <div style="font-weight: bold; color: #856404; margin-bottom: 10px;">
                ⚠️ JSON Parse Error: {str(e)}
            </div>
            <pre style="background: #fff; padding: 15px; border-radius: 4px; border: 1px solid #e0e0e0; 
                       overflow-x: auto; font-family: 'Courier New', monospace; font-size: 0.9em; 
                       line-height: 1.4; color: #333; margin: 0; white-space: pre-wrap; 
                       word-wrap: break-word;">{html.escape(str(data))}</pre>
        </div>
        """
    except Exception as e:
        # Handle any other errors
        return f"""
        <div style="border: 1px solid #dc3545; border-radius: 4px; padding: 10px; background: #f8d7da;">
            <div style="font-weight: bold; color: #721c24; margin-bottom: 10px;">
                ❌ Error: {str(e)}
            </div>
            <pre style="background: #fff; padding: 15px; border-radius: 4px; border: 1px solid #e0e0e0; 
                       overflow-x: auto; font-family: 'Courier New', monospace; font-size: 0.9em; 
                       line-height: 1.4; color: #333; margin: 0; white-space: pre-wrap; 
                       word-wrap: break-word;">{html.escape(str(data))}</pre>
        </div>
        """



def prepare_message_for_gradio(message):
    """
    Dynamically format a message object into HTML for better visualization
    """
    msg_str = str(message)
    
    # Extract message attributes dynamically using more robust parsing
    attributes = {}
    
    # Try to extract source
    source_match = re.search(r"source='([^']*)'", msg_str)
    if source_match:
        attributes['source'] = source_match.group(1)
    
    # Try to extract type
    type_match = re.search(r"type='([^']*)'", msg_str)
    if type_match:
        attributes['type'] = type_match.group(1)
    
    # Try to extract models_usage
    usage_match = re.search(r"models_usage=([^,\s]+)", msg_str)
    if usage_match:
        attributes['models_usage'] = usage_match.group(1)
    
    # Extract content more carefully - handle both quoted strings and complex objects
    content = msg_str

    # Fallback: Extract everything between content= and type= (if exists)
    if 'content' not in attributes and 'content=' in msg_str:
        content_start = msg_str.find('content=') + len('content=')

        # Determine end of content
        type_marker_index = msg_str.find(' type=', content_start)
        content_end = type_marker_index if type_marker_index != -1 else len(msg_str)

        # Extract raw content string
        raw_content = msg_str[content_start:content_end].strip()

        # Remove surrounding quotes if any
        if (raw_content.startswith("'") and raw_content.endswith("'")) or \
        (raw_content.startswith('"') and raw_content.endswith('"')):
            raw_content = raw_content[1:-1]

        # Unescape escaped characters safely
        attributes['content'] = raw_content.encode('utf-8').decode('unicode_escape')

        
    # Generate HTML based on message type and source
    html_parts = []
    
    # Header with source and type
    source = attributes.get('source', 'Unknown')
    msg_type = attributes.get('type', 'Unknown')
    timestamp = datetime.now().strftime("%H:%M:%S")
    

    if 'Orchestrator' in source:
        source = 'Orchestrator'


    # Enhanced color coding for different sources with emojis
    source_configs = {
        'user': {'color': '#e3f2fd', 'emoji': '👤', 'name': 'User'},
        'Orchestrator': {'color': '#f3e5f5', 'emoji': '🎯', 'name': 'Orchestrator'},
        'GoogleSearcher': {'color': '#e8f5e8', 'emoji': '🔍', 'name': 'Google Search'},
        'PudMedPaperSearcher': {'color': '#fff3e0', 'emoji': '📚', 'name': 'PubMed Search'},
        'ScientistAgent1': {'color': '#fce4ec', 'emoji': '🧬', 'name': 'Scientist1'},
        'FileReader': {'color': '#e0f2f1', 'emoji': '📄', 'name': 'File Reader'}
    }
    
    config = source_configs.get(source, {'color': '#f5f5f5', 'emoji': '🤖', 'name': source})
    bg_color = config['color']
    emoji = config['emoji']
    display_name = config['name']
    
    # Add usage info if available
    usage_info = ""
    if attributes.get('models_usage') and attributes['models_usage'] != 'None':
        try:
            # Try to extract token usage from models_usage
            usage_str = attributes['models_usage']
            tokens_match = re.search(r'prompt_tokens=(\d+)[^,]*completion_tokens=(\d+)', usage_str)
            if tokens_match:
                prompt_tokens = tokens_match.group(1)
                completion_tokens = tokens_match.group(2)
                usage_info = f" | 🔢 {prompt_tokens}→{completion_tokens} tokens"
        except:
            pass
    
    html_parts.append(f"""
    <div class="message-container" style="border: 1px solid #ddd; margin: 10px 0; border-radius: 8px; background-color: {bg_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="background: linear-gradient(135deg, #2a003f, #4b225c); color: white; padding: 8px 12px; border-radius: 8px 8px 0 0; font-weight: bold;">
            <span style="color: #ffd700;">{emoji} {display_name}</span> 
            <span style="float: right; font-size: 0.8em; color: #ccc;">{timestamp} | {msg_type}{usage_info}</span>
        </div>
        <div style="padding: 12px;">
    """)
    
    # Content formatting based on type
    content = attributes.get('content', msg_str)
    
    if msg_type == 'ToolCallRequestEvent' or 'FunctionCall' in content:
        # Format function calls nicely
        html_parts.append('<h4 style="color: #2a003f; margin-top: 0; display: flex; align-items: center;"><span style="margin-right: 8px;">🔧</span> Tool Execution:</h4>')
        
        # Extract function calls with improved regex
        function_calls = re.findall(r"FunctionCall\([^)]*?name='([^']*)'[^)]*?arguments='([^']*?)'[^)]*?\)", content)
        
        if function_calls:
            html_parts.append('<div style="display: grid; gap: 8px;">')
            for i, (name, args) in enumerate(function_calls, 1):
                try:
                    # Try to parse arguments as JSON for better display
                    args_dict = json.loads(args.replace('\\"', '"'))
                    args_display = json.dumps(args_dict, indent=2)
                    args_preview = list(args_dict.keys())[0] if args_dict else "No args"
                except:
                    args_display = args
                    args_preview = args[:50] + "..." if len(args) > 50 else args
                
                html_parts.append(f"""
                <div class="tool-call" style="background: rgba(42, 0, 63, 0.05); border: 1px solid rgba(42, 0, 63, 0.2); border-radius: 6px; padding: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="color: #2a003f; display: flex; align-items: center;">
                            <span style="background: #2a003f; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8em; margin-right: 8px;">{i}</span>
                            {name}
                        </strong>
                        <small style="color: #666; font-style: italic;">{args_preview}</small>
                    </div>
                    <details style="cursor: pointer;">
                        <summary style="color: #4b225c; font-weight: 500;">View Arguments</summary>
                        <pre style="margin: 8px 0 0 0; font-size: 0.85em; background: white; padding: 8px; border-radius: 4px; overflow-x: auto; border: 1px solid #eee;">{args_display}</pre>
                    </details>
                </div>
                """)
            html_parts.append('</div>')
        else:
            html_parts.append(f'<pre style="background: white; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; border: 1px solid #eee;">{content}</pre>')
    
    elif msg_type == 'TextMessage':

        # Format text messages with proper line breaks and structure
        formatted_content = content.replace('\\n', '\n')
        
        # Check if it's structured content (like the orchestrator's plan)
        if any(marker in formatted_content for marker in ['**', '1.', '2.', '3.', '*   ', 'FACTS TO', 'PLAN:']):
            # Convert markdown-like formatting to HTML
            formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #2a003f;">\1</strong>', formatted_content)
            formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
            formatted_content = re.sub(r'^(\d+\..*?)$', r'<h4 style="color: #4b225c; margin: 12px 0 6px 0;">\1</h4>', formatted_content, flags=re.MULTILINE)
            formatted_content = formatted_content.replace('\n', '<br>')
            html_parts.append(f'<div style="line-height: 1.7; font-size: 1.05em;">{formatted_content}</div>')
        else:
            # Regular text content
            formatted_content = formatted_content.replace('\n', '<br>')
            html_parts.append(f'<div style="line-height: 1.6; font-size: 1.05em; color: #333;">{formatted_content}</div>')
        if formatted_content.startswith("##"):
            return source, gr.Markdown(formatted_content)
    elif msg_type == 'ToolCallRequestEvent' or 'FunctionCall' in content:
        # Format function calls nicely
        html_parts.append('<h4 style="color: #2a003f; margin-top: 0; display: flex; align-items: center;"><span style="margin-right: 8px;">🔧</span> Tool Execution:</h4>')
        
        # Extract function calls with improved regex
        function_calls = re.findall(r"FunctionCall\([^)]*?name='([^']*)'[^)]*?arguments='([^']*?)'[^)]*?\)", content)
        
        if function_calls:
            html_parts.append('<div style="display: grid; gap: 8px;">')
            for i, (name, args) in enumerate(function_calls, 1):
                try:
                    # Try to parse arguments as JSON for better display
                    args_dict = json.loads(args.replace('\\"', '"'))
                    args_display = json.dumps(args_dict, indent=2)
                    args_preview = list(args_dict.keys())[0] if args_dict else "No args"
                except:
                    args_display = args
                    args_preview = args[:50] + "..." if len(args) > 50 else args
                
                html_parts.append(f"""
                <div class="tool-call" style="background: rgba(42, 0, 63, 0.05); border: 1px solid rgba(42, 0, 63, 0.2); border-radius: 6px; padding: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="color: #2a003f; display: flex; align-items: center;">
                            <span style="background: #2a003f; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8em; margin-right: 8px;">{i}</span>
                            {name}
                        </strong>
                        <small style="color: #666; font-style: italic;">{args_preview}</small>
                    </div>
                    <details style="cursor: pointer;">
                        <summary style="color: #4b225c; font-weight: 500;">View Arguments</summary>
                        <pre style="margin: 8px 0 0 0; font-size: 0.85em; background: white; padding: 8px; border-radius: 4px; overflow-x: auto; border: 1px solid #eee;">{args_display}</pre>
                    </details>
                </div>
                """)
            html_parts.append('</div>')
        else:
            html_parts.append(f'<pre style="background: white; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; border: 1px solid #eee;">{content}</pre>')

    elif msg_type == 'ToolCallExecutionEvent':
        # FIXED: Use content-only function instead of full HTML document
        html_parts.append('<h4 style="color: #2a003f; margin-top: 0; display: flex; align-items: center;"><span style="margin-right: 8px;">📋</span> Tool Results:</h4>')
        
        content_start = msg_str.find('content=') + 8
        content_end = msg_str.find(' type=') if ' type=' in msg_str else len(msg_str)
        
        
        potential_content = msg_str[content_start:content_end].strip()
        # print(potential_content)
        
        if potential_content.endswith(".csv')\", name='fetch_data_from_database', call_id='', is_error=False)]"):
            # Find the last occurrence of =, space, ' or " before .csv
            match = re.search(r'[=\s\'"]([^=\s\'"]+\.csv)', potential_content)
            if match:
                filename = "cache/upload_file/" + match.group(1)
                return source, gr.File(filename)
        
        file_paths = []

        # print("extracting file paths")
        # if 'file_path' in potential_content:


        #     print('extracting file paths')
        #     # Extract all file paths from the content using key-value style (with colon)
        #     # Loop-based extraction of file paths
        #     file_path_matches = []
        #     search_start = 0
        #     while search_start  < len(potential_content):
        #         if search_start >= len(potential_content):
        #             break
        #         idx = potential_content.find('file_path', search_start)
        #         if idx == -1 or idx >= len(potential_content):
        #             break
        #         # Find the first ':' after 'file_path'
        #         colon_idx = potential_content.find(':', idx)
        #         if colon_idx == -1 or colon_idx >= len(potential_content):
        #             break
        #         # Find the first quote (single or double) after the colon
        #         quote_idx = -1
        #         quote_char = ''
        #         for q in ["'", '"']:
        #             q_idx = potential_content.find(q, colon_idx + 1)
        #             if q_idx != -1 and (quote_idx == -1 or q_idx < quote_idx):
        #                 quote_idx = q_idx
        #                 quote_char = q
        #         if quote_idx == -1 or quote_idx >= len(potential_content):
        #             search_start = colon_idx + 1
        #             continue
        #         # Find the closing quote
        #         end_quote_idx = potential_content.find(quote_char, quote_idx + 1)
        #         if end_quote_idx == -1 or end_quote_idx >= len(potential_content):
        #             search_start = quote_idx + 1
        #             continue


                
        #         # Extract the file path
        #         file_path = potential_content[quote_idx + 1:end_quote_idx]
        #         file_path_matches.append(file_path)
        #         search_start = end_quote_idx + 1

        #     print(len(file_path_matches), "file paths found")
            
            # for file_path in file_path_matches:
            #     if file_path.startswith("cache"):
            #         # remove non letter or digit suffix of file_path after the last dot
            #         file_path = str(re.sub(r'[^a-zA-Z0-9_.-]+$', '', file_path))
            #         file_paths.append(gr.File(file_path, file_types=['file']))




            #         # new_path = re.sub(r'\.(?=.*\.)', '-', file_path)
            #         # shutil.copy(file_path, new_path)
            #         # file_paths.append(gr.File(new_path))
            # if file_paths:
            #     print("File paths found:", file_paths)
            #     return source, file_paths

        if potential_content.startswith("##"):
            return source, gr.Markdown(potential_content)
        try:
            # Use the content-only function that's safe to embed
            search_data = convert_search_results(potential_content)
            search_html = generate_search_results_content(search_data)
            if search_html.startswith("##"):
                return source, gr.Markdown(search_html)
            html_parts.append(search_html)
        except Exception as e:
            # Fallback to simple display if parsing fails
            html_parts.append(f'<div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 8px; border-radius: 4px; color: #856404;">⚠️ Could not parse tool results: {html.escape(str(e))}</div>')
            html_parts.append(f'<pre style="background: white; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; border: 1px solid #eee; max-height: 200px; overflow-y: auto;">{html.escape(potential_content)}</pre>')
    
    else:
        # Default case for unrecognized types
        html_parts.append(f'<div style="color: #333; font-size: 1.05em;">{html.escape(content)}</div>')

    # Close the main div
    html_parts.append('</div></div>')
    
    return source, ''.join(html_parts)

# If you still need the original functionality, here's a compatibility function:
def parse_msg_simple(msg):
    """
    Simple version that maintains the original interface - returns source and msg
    """
    msg_str = str(msg)
    
    # Extract source
    source_match = re.search(r"source='([^']*)'", msg_str)
    source = source_match.group(1) if source_match else "other"
    
    return source, msg_str




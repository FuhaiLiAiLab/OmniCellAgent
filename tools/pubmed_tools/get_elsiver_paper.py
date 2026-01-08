import xml.etree.ElementTree as ET
import os
import re

def extract_article_sections_from_toc(xml_text):
    """
    Extract article sections based on the table of contents in the XML
    
    Parameters:
    - xml_text: XML text of the article
    
    Returns:
    - Dictionary containing sections with their content and subsections
    """
    # Define namespaces used in Elsevier XML
    namespaces = {
        'default': 'http://www.elsevier.com/xml/svapi/article/dtd',
        'ce': 'http://www.elsevier.com/xml/common/dtd',
        'ja': 'http://www.elsevier.com/xml/ja/dtd',
        'xocs': 'http://www.elsevier.com/xml/xocs/dtd',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
        'mml': 'http://www.w3.org/1998/Math/MathML'
    }
    
    def extract_text_from_element(element):
        """
        Recursively extract all text from an element, with special handling for formatting elements
        """
        if element is None:
            return ""
            
        # Get direct text content
        text = element.text or ""
        
        # Process child elements with special handling for formatting
        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            # Special handling for specific elements
            if tag == 'sup':
                # Handle superscript - clean up spaces and add proper notation
                child_text = extract_text_from_element(child)
                # Remove extra spaces in superscripts
                child_text = re.sub(r'\s+', '', child_text)
                text += "^" + child_text
            elif tag == 'sub':
                # Handle subscript
                child_text = extract_text_from_element(child)
                child_text = re.sub(r'\s+', '', child_text)
                text += "_" + child_text
            elif tag == 'italic':
                child_text = extract_text_from_element(child)
                text += child_text
            elif tag == 'bold':
                child_text = extract_text_from_element(child)
                text += child_text
            else:
                # Standard recursive processing for other elements
                text += extract_text_from_element(child)
            
            # Add any tail text (text that follows the child element)
            if child.tail:
                text += child.tail
                
        return text
    
    def find_section_content(section_title, parent_element=None):
        """
        Find the content of a section by its title, with proper escaping for XPath
        
        Parameters:
        - section_title: The title of the section to find
        - parent_element: The parent element to search within (for subsections)
        
        Returns:
        - The section content as a string and the section element
        """
        # First, try a more direct approach without using the section title in XPath
        if parent_element is None:
            # For root level, search all sections and check titles manually
            sections = root.findall('.//ce:section', namespaces)
            search_element = root
        else:
            # For subsections, search within parent
            sections = parent_element.findall('./ce:section', namespaces)
            search_element = parent_element
        
        # Find section by comparing titles
        section_elem = None
        for section in sections:
            title_elem = section.find('./ce:section-title', namespaces)
            if title_elem is not None:
                title_text = extract_text_from_element(title_elem)
                # Compare with the target title
                if title_text == section_title or section_title in title_text:
                    section_elem = section
                    break
        
        if section_elem is None:
            return "", None
        
        # Extract content from paragraphs directly under this section
        content_parts = []
        for para in section_elem.findall('./ce:para', namespaces):
            para_text = extract_text_from_element(para)
            if para_text:
                # Clean up excess whitespace
                para_text = re.sub(r'\s+', ' ', para_text).strip()
                content_parts.append(para_text)
        
        return '\n\n'.join(content_parts), section_elem
    
    def process_toc_entry(toc_entry):
        """
        Process a TOC entry recursively to build the section structure
        
        Parameters:
        - toc_entry: The TOC entry element to process
        
        Returns:
        - A tuple of (section_title, section_data)
        """
        # Get the section title
        title_elem = toc_entry.find('./xocs:item-toc-section-title', namespaces)
        if title_elem is None:
            return None, None
        
        section_title = title_elem.text
        
        # Find section content based on title
        section_content, section_elem = find_section_content(section_title)
        
        # Prepare section data structure
        section_data = {'content': section_content, 'subsections': {}}
        
        # Process subsections recursively
        for subsection_entry in toc_entry.findall('./xocs:item-toc-entry', namespaces):
            subsec_title, subsec_data = process_toc_entry(subsection_entry)
            if subsec_title and subsec_data:
                section_data['subsections'][subsec_title] = subsec_data
        
        return section_title, section_data
    
    try:
        # Parse XML
        root = ET.fromstring(xml_text)
        
        # Find the table of contents
        toc = root.find('.//xocs:item-toc', namespaces)
        if toc is None:
            # Fall back to standard section extraction if no TOC is found
            print("No table of contents found, falling back to standard extraction.")
            return extract_article_sections(xml_text) # This now refers to the fallback function
        
        # Process each top-level TOC entry
        sections = {}
        for toc_entry in toc.findall('./xocs:item-toc-entry', namespaces):
            section_title, section_data = process_toc_entry(toc_entry)
            if section_title and section_data:
                sections[section_title] = section_data
        
        return sections
    
    except Exception as e:
        print(f"Error parsing XML: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_article_sections(xml_text): # Renamed from extract_article_sections to avoid conflict, now the primary fallback
    """
    Fallback function to extract article sections when TOC is not available or for general XML.
    
    Parameters:
    - xml_text: XML text of the article
    
    Returns:
    - Dictionary containing sections with their content and subsections
    """
    # Define namespaces used in Elsevier XML (can be adapted for more general XML if needed)
    namespaces = {
        'default': 'http://www.elsevier.com/xml/svapi/article/dtd',
        'ce': 'http://www.elsevier.com/xml/common/dtd',
        'ja': 'http://www.elsevier.com/xml/ja/dtd',
        'xocs': 'http://www.elsevier.com/xml/xocs/dtd',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
        'mml': 'http://www.w3.org/1998/Math/MathML'
    }
    
    def extract_text_from_element(element):
        """
        Recursively extract all text from an element, with special handling for formatting elements
        """
        if element is None:
            return ""
            
        # Get direct text content
        text = element.text or ""
        
        # Process child elements with special handling for formatting
        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            # Special handling for specific elements
            if tag == 'sup':
                # Handle superscript - clean up spaces and add proper notation
                child_text = extract_text_from_element(child)
                # Remove extra spaces in superscripts
                child_text = re.sub(r'\s+', '', child_text)
                text += "^" + child_text
            elif tag == 'sub':
                # Handle subscript
                child_text = extract_text_from_element(child)
                child_text = re.sub(r'\s+', '', child_text)
                text += "_" + child_text
            elif tag == 'italic':
                child_text = extract_text_from_element(child)
                text += child_text
            elif tag == 'bold':
                child_text = extract_text_from_element(child)
                text += child_text
            else:
                # Standard recursive processing for other elements
                text += extract_text_from_element(child)
            
            # Add any tail text (text that follows the child element)
            if child.tail:
                text += child.tail
                
        return text
    
    def process_section(section_elem):
        """Process a section element including any nested subsections"""
        section_data = {'content': '', 'subsections': {}}
        
        # Get the section title
        title_elem = section_elem.find('./ce:section-title', namespaces)
        section_title = extract_text_from_element(title_elem) if title_elem is not None else "Untitled Section"
        
        # Extract content from paragraphs directly under this section (not in subsections)
        content_parts = []
        for para in section_elem.findall('./ce:para', namespaces):
            # Extract all text from paragraph, including from nested elements
            para_text = extract_text_from_element(para)
            if para_text:
                # Clean up excess whitespace
                para_text = re.sub(r'\s+', ' ', para_text).strip()
                content_parts.append(para_text)
        
        # Combine all paragraph texts
        section_data['content'] = '\n\n'.join(content_parts)
        
        # Process any subsections
        for subsection in section_elem.findall('./ce:section', namespaces):
            subsec_title, subsec_data = process_section(subsection)
            section_data['subsections'][subsec_title] = subsec_data
        
        return section_title, section_data
    
    try:
        # Parse XML
        root = ET.fromstring(xml_text)
        
        # Find main sections
        sections = {}
        
        # Look for sections in the XML, handling various possible paths
        # These paths are Elsevier specific, might need generalization for other XMLs
        section_paths = [
            './/ce:sections/ce:section',
            './/default:originalText/ce:sections/ce:section',
            './/default:fulltext/ce:sections/ce:section',
            './/ce:section'  # Sometimes sections are directly at root level
        ]
        
        found_sections = False
        for path in section_paths:
            for section in root.findall(path, namespaces):
                section_title, section_data = process_section(section)
                sections[section_title] = section_data
                found_sections = True
        
        if not found_sections:
            print("No sections found using common Elsevier paths. Consider adapting section_paths for your XML structure.")
            # Basic attempt: if no ce:section, try to find any direct children that might be sections
            # This is a very generic fallback and might not work well.
            for child_elem in root:
                 # Heuristic: if an element has a 'title' child or attribute, treat it as a section
                title_text = child_elem.get("title") or (child_elem.find("title").text if child_elem.find("title") is not None else None)
                if not title_text: # Try common title tag names
                    for common_title_tag in ['title', 'heading', 'header']:
                        title_node = child_elem.find(f'.//{common_title_tag}', namespaces)
                        if title_node is not None and title_node.text:
                            title_text = title_node.text
                            break
                if not title_text:
                    title_text = child_elem.tag.split('}')[-1] if '}' in child_elem.tag else child_elem.tag


                content_text = extract_text_from_element(child_elem)
                # Avoid adding the whole document as one section if no title found
                if title_text != (child_elem.tag.split('}')[-1] if '}' in child_elem.tag else child_elem.tag) or content_text != extract_text_from_element(root):
                     sections[title_text] = {'content': content_text.strip(), 'subsections': {}}


        return sections
    
    except Exception as e:
        print(f"Error parsing XML: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_sections_for_display(sections, indent_level=0, prefix=""):
    """
    Format sections and subsections for readable plain text display
    with explicit newlines and indentation.
    
    Parameters:
    - sections: Dictionary of sections with their content and subsections
    - indent_level: Current indentation level (number of spaces)
    - prefix: Current numbering prefix (e.g., "1.", "1.1.")
    
    Returns:
    - Formatted plain text string representation of sections and subsections
    """
    result = ""
    section_counter = 0
    indent_str = "  " * indent_level  # Two spaces per indent level

    for title, data in sections.items():
        section_counter += 1
        current_number_prefix = f"{prefix}{section_counter}." if prefix else f"{section_counter}."
        
        # Add section title with appropriate indentation and numbering
        result += f"{indent_str}{current_number_prefix} {title}\n"
        
        # Add section content
        if isinstance(data, dict) and 'content' in data:
            if data['content']:
                # Indent content further than the title
                content_indent_str = "  " * (indent_level + 1)
                # Split content by newlines and indent each line
                for line in data['content'].split('\n'):
                    result += f"{content_indent_str}{line}\n"
                result += "\n"  # Add a blank line after content
            
            # Add subsections if they exist
            if data['subsections']:
                result += format_sections_for_display(data['subsections'], indent_level + 1, prefix=current_number_prefix + " ")
        elif isinstance(data, str): # Handle case where data might be just content string
            # Indent content further than the title
            content_indent_str = "  " * (indent_level + 1)
            for line in data.split('\n'):
                result += f"{content_indent_str}{line}\n"
            result += "\n"  # Add a blank line after content
    
    return result


def get_article_sections_from_xml(xml_text, section_names=None):
    """
    Main entry function: Extract specific sections from XML text.
    
    Parameters:
    - xml_text: The XML content as a string.
    - section_names: List of section names to extract (if None, extract all sections).
    
    Returns:
    - Dictionary containing the requested sections with subsections.
    - None if parsing or extraction failed.
    """
    if not xml_text:
        print("Error: XML text is empty.")
        return None
    
    all_sections = None
    try:
        
        all_sections = extract_article_sections_from_toc(xml_text)
        if not all_sections:
            print("Failed to extract any sections.")
            return None
            
        # If specific sections requested, filter them
        if section_names and all_sections:
            # Ensure all_sections is a dictionary before trying to filter
            if isinstance(all_sections, dict):
                filtered_sections = {name: all_sections.get(name) for name in section_names if name in all_sections}
                if not filtered_sections:
                    print(f"Warning: None of the requested sections {section_names} were found.")
                return filtered_sections
            else:
                print("Warning: all_sections is not a dictionary, cannot filter by section_names.")
                return all_sections # Or handle as an error
        
        return all_sections
        
    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")
        return None
    except Exception as e:
        print(f"Error extracting sections: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    

def save_sections_to_files(sections, output_dir=None, prefix=""):
    """
    Save sections to individual files
    
    Parameters:
    - sections: Dictionary of sections with their content and subsections
    - output_dir: Directory to save files (default: current directory)
    - prefix: Prefix for filenames (for nested sections)
    """
    if output_dir is None:
        output_dir = "article_sections"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not isinstance(sections, dict):
        print(f"Warning: Expected a dictionary of sections, but got {type(sections)}. Cannot save.")
        return

    for title, data in sections.items():
        # Create a safe filename
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        filename = f"{prefix}{safe_title}.txt" if prefix else f"{safe_title}.txt"
        filepath = os.path.join(output_dir, filename)
        
        if isinstance(data, dict) and 'content' in data:
            # Save the section content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"==== {title} ====\n\n")
                f.write(data['content'])
            
            # Process subsections recursively
            if data['subsections']:
                new_prefix = f"{prefix}{safe_title}_" if prefix else f"{safe_title}_"
                save_sections_to_files(data['subsections'], output_dir, new_prefix)
        elif isinstance(data, str): # Handle case where data might be just content string
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"==== {title} ====\n\n")
                f.write(data)
        else:
            print(f"Warning: Section '{title}' has unexpected data format: {type(data)}. Skipping save for this section.")


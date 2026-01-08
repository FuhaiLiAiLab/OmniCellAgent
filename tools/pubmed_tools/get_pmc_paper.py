import xml.etree.ElementTree as ET

def get_infon_text(element, key):
    """Helper to extract text from 'infon' tags with a specific key attribute."""
    for infon in element.findall('infon'):
        if infon.get('key') == key and infon.text:
            return infon.text.strip()
    return None

def parse_xml_to_sections(xml_file_path):
    """
    Parses an XML file and extracts hierarchical content into a formatted plain text string,
    dynamically handling subsections (title_1, title_2, etc.) with indentation, 
    stopping at the first reference section.
    
    Args:
        xml_file_path (str): Path to the XML file.
        
    Returns:
        str: A formatted string with titles and text content, or None on error.
    """
    output_string = ""
    
    # Counters and state for dynamic hierarchical numbering
    main_section_abs_counter = 0  # Absolute counter for main sections (1, 2, 3...)
    current_subsection_levels = [] # Stores counts for sub-levels e.g. [1] for X.1, [1,1] for X.1.1
    
    # Track current main section context
    current_main_section_type_str = None # e.g., "INTRO", "METHODS"
    last_added_was_title = False

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        for passage in root.findall('.//document/passage'):
            # Get section type (used for parent section tracking)
            xml_section_type = get_infon_text(passage, "section_type")
            if not xml_section_type:
                continue  # Skip passages without section_type
                
            if xml_section_type == "REF":
                # print(f"Encountered REF section, stopping processing.") # Optional
                break
                
            passage_type = get_infon_text(passage, "type")
            text_element = passage.find('text')
            text_content = text_element.text.strip() if text_element is not None and text_element.text else ""
            
            # Skip empty paragraphs. Titles can be empty (structural).
            if passage_type == "paragraph" and not text_content:
                continue
            # If no type and no text, also skip.
            if not passage_type and not text_content and not (passage_type and passage_type.startswith("title_")): # Allow empty title_X
                continue

            current_indent_level = 0

            # Check if we've entered a new main section (based on xml_section_type changing)
            if xml_section_type != current_main_section_type_str:
                main_section_abs_counter += 1
                current_subsection_levels = []  # Reset subsection numbering
                current_main_section_type_str = xml_section_type
                current_indent_level = 0
                
                main_section_title_text = current_main_section_type_str.lower()
                # If this passage itself is the main title (heuristic: no type or type="title")
                if (not passage_type or passage_type == "title") and text_content:
                    main_section_title_text = text_content.lower()

                formatted_title = f"{main_section_abs_counter}. {main_section_title_text}"
                if output_string and not output_string.endswith("\\n\\n"):
                    output_string += "\\n" # Add extra newline before new main section if needed
                output_string += f"{formatted_title}\\n"
                last_added_was_title = True

                # If this passage's content was used for the main title, or it was an empty structural title,
                # consume it and continue to the next passage.
                if (not passage_type or passage_type == "title"):
                    continue 
            
            # Handle passage based on its type (title_X, paragraph, or updating main section title)
            if passage_type and passage_type.startswith("title_"):
                try:
                    level = int(passage_type.split('_')[1])
                    if level <= 0: raise ValueError("Level must be positive")
                except (ValueError, IndexError): # Invalid title_X format, treat as paragraph
                    if text_content:
                        indent = "  " * (len(current_subsection_levels) + 1) # Indent based on last known level
                        if not last_added_was_title: output_string += "\\n" # Add newline if previous was text
                        output_string += f"{indent}{text_content}\\n"
                        last_added_was_title = False
                    continue 

                current_indent_level = level
                # Adjust current_subsection_levels for the new title's depth
                if level > len(current_subsection_levels):
                    current_subsection_levels.extend([1] * (level - len(current_subsection_levels)))
                else: 
                    current_subsection_levels = current_subsection_levels[:level] 
                    current_subsection_levels[level-1] += 1 
                
                title_number_prefix = str(main_section_abs_counter)
                if current_subsection_levels:
                    title_number_prefix += "." + '.'.join(map(str, current_subsection_levels[:level])) # Use level for prefix
                
                indent = "  " * current_indent_level
                formatted_title = f"{indent}{title_number_prefix} {text_content}"
                if output_string and not output_string.endswith("\\n\\n") and not last_added_was_title :
                     output_string += "\\n" 
                output_string += f"{formatted_title}\\n"
                last_added_was_title = True
            
            elif passage_type == "paragraph":
                if text_content:
                    indent_level_for_text = current_indent_level + 1 if current_subsection_levels else 1
                    indent = "  " * indent_level_for_text
                    # Add newline before paragraph if the last thing added was also text, to separate paragraphs
                    if not last_added_was_title and output_string and not output_string.endswith("\\n\\n") and not output_string.endswith(indent):
                        output_string += "\\n"
                    elif last_added_was_title and output_string and not output_string.endswith("\\n"): # Ensure newline after title
                         output_string += "\\n"

                    output_string += f"{indent}{text_content}\\n"
                    last_added_was_title = False

            elif not passage_type and text_content: 
                # This could be the actual title for the main section if it was set generically.
                # Attempt to replace the generic main section title
                generic_title_to_replace = f"{main_section_abs_counter}. {current_main_section_type_str.lower()}\\n"
                if output_string.endswith(generic_title_to_replace):
                    output_string = output_string[:-len(generic_title_to_replace)] # Remove generic title
                    output_string += f"{main_section_abs_counter}. {text_content.lower()}\\n" # Add specific title
                    last_added_was_title = True
                elif text_content: # Otherwise, append as content to the last section
                    indent = "  " * 1 # Default indent for unclassified text under main section
                    if not last_added_was_title: output_string += "\\n"
                    output_string += f"{indent}{text_content}\\n"
                    last_added_was_title = False
            
            elif text_content: # Catch-all for other types with text not handled above
                is_processed_type = passage_type in ["title", "paragraph"] or \
                                    (passage_type and passage_type.startswith("title_"))
                if not is_processed_type:
                    indent_level_for_text = current_indent_level + 1 if current_subsection_levels else 1
                    indent = "  " * indent_level_for_text
                    if not last_added_was_title: output_string += "\\n"
                    output_string += f"{indent}{text_content}\\n"
                    last_added_was_title = False
                        
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found at {xml_file_path}")
        return None
        
    return output_string.strip() # Remove any leading/trailing whitespace from the final string
